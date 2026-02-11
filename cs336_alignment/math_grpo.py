# math_grpo.py
import os
import torch
import wandb
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
import cs336_alignment.grpo_helpers as grpo_helpers
import cs336_alignment.sft_helpers as sft_helpers
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from vllm import SamplingParams
from contextlib import nullcontext
import typer
from omegaconf import OmegaConf
from cs336_alignment.configs import grpo_config

app = typer.Typer(add_completion=False)

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def train_loop(
    ctx: typer.Context,
    config: Path = typer.Option(None, "--config", exists=True, help="YAML/JSON config file"),
    save_resolved: bool = typer.Option(True, help="Save resolved config in output_dir"),
):
    # build config
    overrides = list(ctx.args)
    cfg = grpo_config.build_cfg(config, overrides)
    cfg.io.output_dir.mkdir(parents=True, exist_ok=True)
    if save_resolved:
        # Save as YAML with all values resolved
        OmegaConf.save(OmegaConf.structured(cfg), cfg.io.output_dir / "config.resolved.yaml")
    typer.echo(OmegaConf.to_yaml(OmegaConf.structured(cfg)))

    # sanity check
    m_t_bs, n_pprb, n_mbprb = grpo_config.cfg_sanity_check(cfg)
    print(f"Micro train batch size: {m_t_bs}, number of prompts per rollout batch: {n_pprb}, number of microbatches per rollout batch: {n_mbprb}")
    gs = cfg.grpo.group_size
    rollout_bs = cfg.grpo.rollout_batch_size
    n_grpo_steps = cfg.grpo.n_grpo_steps
    epochs_per_rollout_batch = cfg.train.epochs_per_rollout_batch
    device_train = cfg.model.device_train
    device_eval = cfg.model.device_eval
    grad_acc_steps = cfg.train.gradient_accumulation_steps

    # set seed
    torch.manual_seed(cfg.train.seed)    # load dataset and construct dataloaders
    train_ds_formatted, valid_ds_formatted = sft_helpers.load_gsm8k_dataset(cfg.train.max_train_samples, cfg.train.max_eval_samples, cfg.train.seed)
    print("Loaded train and valid datasets, with lengths: ", len(train_ds_formatted), len(valid_ds_formatted))
    
    # load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_id,
        #torch_dtype=torch.bfloat16,
        #attn_implementation="flash_attention_2",
        ).to(device_train)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_id)
    print("Model and tokenizer loaded to device: ", device_train)

    # initialize vLLM instance
    llm = sft_helpers.init_vllm(cfg.model.model_id, device_eval, cfg.train.seed)
    print("vLLM instance initialized to device: ", device_eval)

    # initialize sampling parameters
    sampling_params = SamplingParams(
        temperature=cfg.sampling.temperature,
        top_p=cfg.sampling.top_p,
        max_tokens=cfg.sampling.max_tokens,
        min_tokens=cfg.sampling.min_tokens,
        stop=cfg.sampling.stop,
        include_stop_str_in_output=True,
    )

    # initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.optim.lr,
        betas=cfg.optim.betas,
        weight_decay=cfg.optim.weight_decay,
    )
    print("Optimizer initialized")
    
    # initialize wandb
    wandb.init(
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        name=cfg.wandb.name,
        mode=cfg.wandb.mode,
        dir=cfg.io.output_dir,
    )
    print("Wandb initialized")

    global_step = 0
    best_reward = -float('inf')
    # train loop
    for grpo_step in range(n_grpo_steps):
        print(f"Starting GRPO step {grpo_step} of {n_grpo_steps}")
        # rollout batch
        shuffle_idx = torch.randperm(len(train_ds_formatted)).tolist()[:n_pprb]
        prompts = [train_ds_formatted["prompt"][i] for i in shuffle_idx for _ in range(gs)]
        gt_answers = [train_ds_formatted["answer"][i] for i in shuffle_idx for _ in range(gs)]
        grouped_outputs = llm.generate(
            prompts, 
            sampling_params,
        )
        rollout_responses = [output.text for group in grouped_outputs for output in group.outputs]
        print(f"Finished rolling out batch of {len(rollout_responses)} responses")

        # compute rewards
        advantages, raw_rewards, metadata = grpo_helpers.compute_group_normalized_rewards(
            r1_zero_reward_fn,
            rollout_responses, 
            gt_answers, 
            gs, 
            cfg.grpo.advantage_eps,
            normalize_by_std=cfg.grpo.use_std_normalization,
        )
        format_rewards = metadata["format_reward"].to(device_train)
        print(f"Finished computing rewards, raw rewards: {raw_rewards.mean().item()}, format rewards: {format_rewards.mean().item()}")

        # reshape advantages, raw_rewards to (rollout_batch_size, 1)
        advantages = advantages.unsqueeze(-1).to(device_train)
        raw_rewards = raw_rewards.unsqueeze(-1).to(device_train)

        # tokenize input and labels, and construct response mask
        tok = sft_helpers.tokenize_prompt_and_output(prompts, rollout_responses, tokenizer)
        input_ids = tok["input_ids"].to(device_train)
        labels = tok["labels"].to(device_train)
        response_mask = tok["response_mask"].to(device_train)
        print(f"Finished tokenizing input and labels, input_ids: {input_ids.shape}, labels: {labels.shape}, response_mask: {response_mask.shape}")

        # compute old policy log probs
        old_policy_log_probs = torch.zeros_like(input_ids, dtype=torch.float32).to(device_train)
        model.eval()
        with torch.inference_mode():
        #with torch.no_grad(): # do not compute gradients for old policy log probs
            # size too large, we need to split into chunks
            for idx in range(0, len(input_ids), m_t_bs):
                mb_idx = slice(idx, idx+m_t_bs)
                mb_input_ids = input_ids[mb_idx]
                mb_labels = labels[mb_idx]
                mb_old_policy_log_probs = sft_helpers.get_response_log_probs(model, mb_input_ids, mb_labels)['log_probs']
                old_policy_log_probs[mb_idx] = mb_old_policy_log_probs.detach()
        model.train()
        print(f"Finished computing old policy log probs")
        print(f"Old policy log probs: {old_policy_log_probs.dtype}")

        # on/off policy training
        step = 0
        optimizer.zero_grad(set_to_none=True)
        train_stats_tracker = {
            "loss": 0.0,
            "grad_norm": 0.0,
            "token_entropy": 0.0,
            "token_count": 0,
            "format_reward": 0.0,
            "reward": 0.0,
            "advantage": 0.0,
            "clip_fraction": 0.0,
        }
        for epoch in range(epochs_per_rollout_batch):
            print(f"Starting epoch {epoch + 1} of {epochs_per_rollout_batch}")
            # shuffle data? shuffule samples or groups? Or not at all?
            if cfg.train.shuffle:
                if cfg.train.shuffle_groups:
                    shuffle_idx = grpo_helpers.shuffle_groups(rollout_bs, gs).to(device_train)
                else:
                    shuffle_idx = torch.randperm(rollout_bs).to(device_train)
            else:
                shuffle_idx = torch.arange(rollout_bs).to(device_train)

            for i in range(n_mbprb):
                global_step += 1
                step += 1

                mb_idx = shuffle_idx[i*m_t_bs:(i+1)*m_t_bs]
                train_stats_tracker["format_reward"] += format_rewards[mb_idx].mean().item()
                train_stats_tracker["reward"] += raw_rewards[mb_idx].mean().item()

                out = sft_helpers.get_response_log_probs(
                    model, 
                    input_ids[mb_idx], 
                    labels[mb_idx],
                    return_token_entropy=True,
                )
                log_probs = out["log_probs"]

                train_stats_tracker["token_entropy"] += (out["token_entropy"] * response_mask[mb_idx]).sum().item()
                train_stats_tracker["token_count"] += response_mask[mb_idx].sum().item()

                loss, metadata = grpo_helpers.grpo_microbatch_train_step(
                    log_probs,
                    response_mask[mb_idx],
                    grad_acc_steps,
                    cfg.grpo.loss_type,
                    raw_rewards[mb_idx],
                    advantages[mb_idx],
                    old_policy_log_probs[mb_idx],
                    cfg.grpo.cliprange,
                    cfg.grpo.normalize_by_length,
                )

                train_stats_tracker["loss"] += metadata["loss_for_logging"]
                if cfg.grpo.loss_type == "grpo_clip":
                    train_stats_tracker["clip_fraction"] += metadata["clip_fraction"]

                if step % grad_acc_steps == 0:
                    # optimizer step
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.max_grad_norm)
                    train_stats_tracker["grad_norm"] += grad_norm.item()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    grad_steps = step // grad_acc_steps
                    if grad_steps % cfg.train.log_grad_steps == 0:
                        # log training stats
                        train_stats = {
                            "train/loss": train_stats_tracker["loss"],
                            "train/grad_norm": train_stats_tracker["grad_norm"] / grad_steps,
                            "train/token_entropy": train_stats_tracker["token_entropy"] / train_stats_tracker["token_count"],
                            "train/format_reward": train_stats_tracker["format_reward"] / grad_acc_steps / grad_steps,
                            "train/reward": train_stats_tracker["reward"] / grad_acc_steps / grad_steps,
                            "train/clip_fraction": train_stats_tracker["clip_fraction"] / grad_acc_steps / grad_steps if cfg.grpo.loss_type == "grpo_clip" else 0.0,
                        }
                        print(f"Logging training stats: {train_stats}")
                        wandb.log(
                            train_stats, 
                            step = global_step,
                        )
                        train_stats_tracker = {
                            "loss": 0.0,
                            "grad_norm": 0.0,
                            "token_entropy": 0.0,
                            "token_count": 0,
                            "format_reward": 0.0,
                            "reward": 0.0,
                            "advantage": 0.0,
                            "clip_fraction": 0.0,
                        }
            print(f"Epoch {epoch + 1} of {epochs_per_rollout_batch} completed")
            
        print(f"GRPO step {grpo_step + 1} of {n_grpo_steps} completed")
        # load policy into vLLM instance
        sft_helpers.load_policy_into_vllm_instance(model, llm)
        print("Loaded policy into vLLM instance")
        # evaluate
        if grpo_step == 0 or (grpo_step + 1) % cfg.grpo.eval_interval == 0:
            print(f"Evaluating on {len(valid_ds_formatted)} examples")
            stats = sft_helpers.log_generations(
                valid_ds_formatted["prompt"], 
                valid_ds_formatted["answer"], 
                llm, 
                model, 
                tokenizer, 
                sampling_params, 
                r1_zero_reward_fn, 
                batch_size=m_t_bs,
                output_path=f"{cfg.io.output_dir}/grpo_step_{grpo_step + 1}_generations.jsonl",
            )
            eval_stats = {
                "eval/average_format_reward": stats["average_format_reward"],
                "eval/average_reward": stats["average_reward"],
                "eval/count_format_0": stats["count_format_0"],
                "eval/count_format_1_answer_0": stats["count_format_1_answer_0"],
                "eval/count_format_1_answer_1": stats["count_format_1_answer_1"],
                "eval/grpo_step": grpo_step + 1,
            }
            print(f"Evaluation stats: {eval_stats}")
            wandb.log(
                eval_stats, 
                step=global_step,
            )

            # save best model
            if eval_stats["eval/average_reward"] > best_reward:
                best_reward = eval_stats["eval/average_reward"]
                model.save_pretrained(f"{cfg.io.output_dir}/best_model")
                tokenizer.save_pretrained(f"{cfg.io.output_dir}/best_model")
                print(f"Saved best model to {cfg.io.output_dir}/best_model")
        
if __name__ == "__main__":
    app()
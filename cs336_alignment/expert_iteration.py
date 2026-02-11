import os
import argparse
import torch
import wandb
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import cs336_alignment.sft_helpers as sft_helpers
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from vllm import SamplingParams

def load_gsm8k_dataset():
    parent_dir = Path(__file__).resolve().parent.parent
    ds = load_dataset(f"{parent_dir}/data/gsm8k")
    ds = ds.map(sft_helpers.to_r1_zero, remove_columns=ds["train"].column_names)
    return ds['train'], ds['test']

def generate_expert_iteration_dataset(
    ds,
    policy_model,
    sampling_params,
    reward_fn,
) -> Dataset:
    model_responses = policy_model.generate(ds['prompt'], sampling_params)
    prompts = []
    completions = []
    for i, (prompt, ground_truth_answer, model_response) in enumerate(zip(ds['prompt'], ds['answer'], model_responses)):
        for response in model_response.outputs:
            reward = reward_fn(response.text, ground_truth_answer)
            if reward['reward'] == 0:
                continue
            prompts.append(prompt)
            completions.append(response.text)
    return Dataset.from_dict({
        'prompt': prompts,
        'completion': completions,
    })

def main(args):
    torch.manual_seed(args.seed)
    # load dataset and construct dataloaders
    train_ds, valid_ds = load_gsm8k_dataset()
    print("Loaded train and valid datasets")

    # load model and tokenizer
    device_train = "cuda:1" 
    device_eval = "cuda:0" # Somehow setting vllm device to 1 use more cuda blocks and memory, and cause memory issues.
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device_train)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("Model and tokenizer loaded to device: ", device_train)

    # initialize vLLM instance
    llm = sft_helpers.init_vllm(model_id, device_eval, args.seed)
    print("vLLM instance initialized to device: ", device_eval)

    # initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)#, betas=(0.9, 0.95), weight_decay=0.01)

    # sampling parameters
    ei_sampling_params = SamplingParams(
        n=args.num_rollouts,
        temperature=1.0,
        top_p=1.0,
        max_tokens=512,
        min_tokens=4,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        #seed=args.seed,
    )
    eval_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=args.seed,
    )

    # initialize wandb
    wandb.init(
        project="cs336_alignment", 
        group=f"expert_iteration_exps_rerun_remove_sampling_seed",
        name=f"{args.ei_batch_size}_ei_batch_{args.num_rollouts}_rollouts_{args.n_sft_epochs}_sft_epochs",
        config=args,
        mode='offline',
        dir=args.output_dir,
    )
    print("Wandb initialized")

    eval_ds_tok = sft_helpers.tokenize_prompt_and_output(valid_ds['prompt'][:10], valid_ds['completion'][:10], tokenizer)
    eval_input_ids, eval_labels, eval_response_mask = eval_ds_tok["input_ids"].to(device_train), eval_ds_tok["labels"].to(device_train), eval_ds_tok["response_mask"].to(device_train)      

    global_step = 0
    best_reward = -float('inf')
    for ei_step in range(args.n_ei_steps):
        sample_indices = torch.randperm(len(train_ds))[:args.ei_batch_size]
        ds_sampled = train_ds.select(sample_indices)
        ds_sft= generate_expert_iteration_dataset(ds_sampled, llm, ei_sampling_params, r1_zero_reward_fn)
        print(f"Expert iteration step {ei_step}:")
        print("EI batch size: ", len(ds_sampled))
        print("SFT sample size: ", len(ds_sft))
        # run sft on filtered dataset
        m_bs = args.microbatch_size
        grad_acc_steps = args.gradient_accumulation_steps
        bs = m_bs * grad_acc_steps
        if len(ds_sft) < bs:
            print(f"Less than {bs} samples after expert iteration")
            grad_acc_steps = len(ds_sft) // m_bs
            if grad_acc_steps == 0:
                raise ValueError(f"Too few samples, please increase ei-batch-size")
            bs = m_bs * grad_acc_steps
            print(f"Reset gradient accumulation steps to {grad_acc_steps} and batch size to {bs}")
        
        n_sft_steps = len(ds_sft) // bs * grad_acc_steps
        print(f"Microbatch size: {args.microbatch_size}, gradient accumulation steps: {grad_acc_steps}, number of SFT steps: {n_sft_steps}")
        sft_step = 0
        loss_for_logging = 0.0
        token_entropy = 0.0
        token_count = 0
        for sft_epoch in range(args.n_sft_epochs):
            # shuffle dataset
            perm = torch.randperm(len(ds_sft)).tolist()
            ds_sft_shuffled = ds_sft.select(perm)
            for i in range(0, n_sft_steps * m_bs, m_bs):
                global_step += 1
                sft_step += 1                
                # sample a batch
                ds_batch = ds_sft_shuffled.select(range(i, i + m_bs))
                ds_tok = sft_helpers.tokenize_prompt_and_output(list(ds_batch['prompt']), list(ds_batch['completion']), tokenizer)
                #print("max seq length: ", ds_tok["input_ids"].shape[1])
                torch.cuda.empty_cache()
                input_ids, labels, response_mask = ds_tok["input_ids"].to(device_train), ds_tok["labels"].to(device_train), ds_tok["response_mask"].to(device_train)
                output = sft_helpers.get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
                token_entropy += (output['token_entropy'] * response_mask).sum().item()
                token_count += response_mask.sum().item()
                policy_log_probs = output['log_probs']
                loss, _ = sft_helpers.sft_microbatch_train_step(policy_log_probs, response_mask, grad_acc_steps)
                loss_for_logging += loss.item()
                if sft_step % grad_acc_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                
                if sft_step % grad_acc_steps == 0:
                    average_token_entropy = token_entropy / token_count
                    token_entropy = 0.0
                    token_count = 0
                    print(f"Global step {global_step}, training loss: {loss_for_logging}, average token entropy: {average_token_entropy}")
                    wandb.log({"train/loss": loss_for_logging, "train/average_token_entropy": average_token_entropy}, step=global_step)
                    loss_for_logging = 0.0
                    
                    # check val loss on first 10 samples
                    model.eval()
                    with torch.no_grad():
                        eval_policy_log_probs = sft_helpers.get_response_log_probs(model, eval_input_ids, eval_labels)['log_probs']
                        eval_loss_for_logging = -sft_helpers.masked_normalize(eval_policy_log_probs, eval_response_mask, dim=-1).mean().item()
                    print(f"Global step {global_step}, validation loss on first 10 samples: {eval_loss_for_logging}")
                    wandb.log({"eval_10_examples/loss": eval_loss_for_logging}, step=global_step)
                    model.train()
            print(f"SFT epoch {sft_epoch} of {args.n_sft_epochs} completed")

        print(f"Expert iteration step {ei_step} completed")
        sft_helpers.cuda_mem_snapshot(device_train)
        torch.cuda.empty_cache()
        sft_helpers.cuda_mem_snapshot(device_train)
        # load policy into vllm instance
        sft_helpers.load_policy_into_vllm_instance(model, llm) 
        print("Policy loaded into vLLM instance")
        # log generations
        eval_stats = sft_helpers.log_generations(
            valid_ds['prompt'], valid_ds['answer'], llm, model, tokenizer, eval_sampling_params, r1_zero_reward_fn, 
        f"{args.output_dir}/ei_{ei_step}_generations.jsonl", args.microbatch_size * 2)
        print(eval_stats)
        eval_stats["ei_step"] = ei_step
        wandb.log(eval_stats, step=global_step)
        print(f"Expert iteration step {ei_step} completed")
        # save checkpoint, skip to save disk space
        # model.save_pretrained(f"{args.output_dir}/ei_{ei_step}_epoch_{sft_epoch}_checkpoint")
        # tokenizer.save_pretrained(f"{args.output_dir}/ei_{ei_step}_epoch_{sft_epoch}_checkpoint")
        # print(f"Saved checkpoint to {args.output_dir}/ei_{ei_step}_epoch_{sft_epoch}_checkpoint")
        # save best model
        if eval_stats["average_reward"] > best_reward:
            best_reward = eval_stats["average_reward"]
            model.save_pretrained(f"{args.output_dir}/best_model")
            tokenizer.save_pretrained(f"{args.output_dir}/best_model")
            print(f"Saved best model to {args.output_dir}/best_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int, required=False)
    parser.add_argument("--n-ei-steps", default=5, type=int, required=False)
    parser.add_argument("--ei-batch-size", default=256, type=int, required=False)
    parser.add_argument("--num-rollouts", default=4, type=int, required=False)
    parser.add_argument("--n-sft-epochs", default=2, type=int, required=False)
    parser.add_argument("--microbatch-size", default=8, type=int, required=False)
    parser.add_argument("--learning-rate", default=3e-5, type=float, required=False)
    parser.add_argument("--gradient-accumulation-steps", default=8, type=int, required=False)
    parser.add_argument("--max-grad-norm", default=1.0, type=float, required=False)
    parent_dir = Path(__file__).resolve().parent.parent
    parser.add_argument("--output-dir", default=f"{parent_dir}/exps/ei_math/test_run", type=str, required=False)
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    main(args)
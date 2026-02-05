import os
import argparse
import torch
import wandb
from pathlib import Path
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
import cs336_alignment.sft_helpers as sft_helpers
from cs336_alignment.math_baseline import to_r1_zero, r1_zero_reward_fn
from vllm import SamplingParams

from contextlib import nullcontext

def load_gsm8k_dataset(microbatch_size, max_train_samples, max_eval_samples, seed):
    parent_dir = Path(__file__).resolve().parent.parent
    ds = load_dataset(f"{parent_dir}/data/gsm8k")
    ds = ds.map(to_r1_zero, remove_columns=ds["train"].column_names)
    g = torch.Generator()
    g.manual_seed(seed)
    if max_train_samples is not None:
        train_ds = ds["train"].shuffle(seed=seed).select(range(max_train_samples))
        print(train_ds)
    else:
        train_ds = ds["train"]
    train_dataloader = DataLoader(train_ds, batch_size=microbatch_size, shuffle=True, generator=g)
    if max_eval_samples is not None:
        valid_ds = ds["test"].shuffle(seed=seed).select(range(max_eval_samples))
    else:
        valid_ds = ds["test"]
    return train_dataloader, valid_ds

def main(args):
    torch.manual_seed(args.seed)
    parent_dir = Path(__file__).resolve().parent.parent
    # load dataset and construct dataloaders
    train_dataloader, valid_ds = load_gsm8k_dataset(args.microbatch_size, args.max_train_samples, args.max_eval_samples, args.seed)
    print(train_dataloader)
    print(len(train_dataloader))
    # load model and tokenizer
    device_train = "cuda:1" 
    device_eval = "cuda:0" # Somehow setting vllm device to 1 use more cuda blocks and memory, and cause memory issues.
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    #loss stops decreasing quickly when using bfloat16.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        #torch_dtype=torch.bfloat16, 
        #attn_implementation="flash_attention_2",
    ).to(device_train)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("Model and tokenizer loaded to device: ", device_train)

    
    llm = sft_helpers.init_vllm(model_id, device_eval, args.seed)
    print("vLLM instance initialized to device: ", device_eval)


    max_epochs = args.max_epochs
    max_sft_steps = max_epochs * len(train_dataloader)

    # initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)#, betas=(0.9, 0.95), weight_decay=0.01)
    scheduler = get_scheduler(
        "cosine",
        optimizer, 
        num_warmup_steps=10, 
        num_training_steps=max_sft_steps // args.gradient_accumulation_steps,
    )
    eval_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    # initialize wandb
    wandb.init(
        project="cs336_alignment", 
        group="math_sft",
        name=f"math_sft_{args.max_train_samples}_samples_bs_{args.microbatch_size * args.gradient_accumulation_steps}_lr_{args.learning_rate}",
        config=args,
        mode="offline",
        dir=f"{args.output_dir}/wandb",
    )

    step = 0
    best_reward = -float("inf")
    for epoch in range(max_epochs):
        for batch in train_dataloader:
            step += 1
            ## mixed precision training
            autocast_context = nullcontext() #torch.autocast(device_type=device_train, dtype=torch.bfloat16)
            with autocast_context:
                tokenized_data = sft_helpers.tokenize_prompt_and_output(batch["prompt"], batch["completion"], tokenizer)
                input_ids, labels, response_mask = tokenized_data["input_ids"].to(device_train), tokenized_data["labels"].to(device_train), tokenized_data["response_mask"].to(device_train)
                policy_log_probs = sft_helpers.get_response_log_probs(model, input_ids, labels)['log_probs']
                average_response_length = response_mask.sum(dim=-1).float().mean().item()
                loss, metadata = sft_helpers.sft_microbatch_train_step(policy_log_probs, response_mask, 
                    args.gradient_accumulation_steps, normalize_constant=average_response_length)
                if step % args.gradient_accumulation_steps == 0:
                    scheduler.step()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                print(f"Step {step} of {max_sft_steps}, loss: {loss.item()}")
                wandb.log({"train/loss": loss.item()}, step=step)
            # check first 10 test examples
            model.eval()
            with torch.no_grad():
                eval_tokenized_data = sft_helpers.tokenize_prompt_and_output(valid_ds["prompt"][:10], valid_ds["completion"][:10], tokenizer)
                eval_input_ids, eval_labels, eval_response_mask = eval_tokenized_data["input_ids"].to(device_train), eval_tokenized_data["labels"].to(device_train), eval_tokenized_data["response_mask"].to(device_train)
                eval_policy_log_probs = sft_helpers.get_response_log_probs(model, eval_input_ids, eval_labels)['log_probs']
                eval_loss = -sft_helpers.masked_normalize(eval_policy_log_probs, eval_response_mask, normalize_constant=eval_response_mask.sum().item())
                print(f"Check validation loss on first 10 examples: {eval_loss.item()}")
                wandb.log({"eval_10_examples/loss": eval_loss.item()}, step=step)
            model.train()
        print(f"Evaluating epoch {epoch}")
        sft_helpers.load_policy_into_vllm_instance(model, llm)
        eval_stats = sft_helpers.log_generations(
            valid_ds["prompt"], valid_ds["answer"], llm, model, tokenizer, eval_sampling_params, r1_zero_reward_fn, 
            f"{args.output_dir}/epoch_{epoch}_generations.jsonl", args.microbatch_size * 2)
        eval_stats["epoch"] = epoch
        wandb.log(eval_stats, step=step)
        print(eval_stats)
        print(f"Epoch {epoch} completed")

        # save checkpoint
        model.save_pretrained(f"{args.output_dir}/epoch_{epoch}_checkpoint")
        tokenizer.save_pretrained(f"{args.output_dir}/epoch_{epoch}_checkpoint")
        print(f"Saved checkpoint to {args.output_dir}/epoch_{epoch}_checkpoint")
        # save best model
        if eval_stats["average_reward"] > best_reward:
            best_reward = eval_stats["average_reward"]
            model.save_pretrained(f"{args.output_dir}/best_model")
            tokenizer.save_pretrained(f"{args.output_dir}/best_model")
            print(f"Saved best model to {args.output_dir}/best_model")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int, required=False)
    parser.add_argument("--max-epochs", default=5, type=int, required=False)
    parser.add_argument("--microbatch-size", default=8, type=int, required=False)
    parser.add_argument("--learning-rate", default=1e-5, type=float, required=False)
    parser.add_argument("--gradient-accumulation-steps", default=1, type=int, required=False)
    parser.add_argument("--max-grad-norm", default=1.0, type=float, required=False)
    parser.add_argument("--max-train-samples", default=None, type=int, required=False, help="If not None, use a subset of the train dataset of this size")
    parser.add_argument("--max-eval-samples", default=None, type=int, required=False, help="If not None, use a subset of the eval dataset of this size")
    parent_dir = Path(__file__).resolve().parent.parent
    parser.add_argument("--output-dir", default=f"{parent_dir}/exps/math_sft", type=str, required=False)
    args = parser.parse_args()
    # Avoid creating dirs at filesystem root (e.g. when $exp_dir is unset)
    if args.output_dir.startswith("/") and args.output_dir.count("/") <= 2:
        parent_dir = Path(__file__).resolve().parent.parent
        args.output_dir = str(parent_dir / "exps" / "math_sft")
        log.info("output-dir looked like a root path; using default: %s", args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    main(args)


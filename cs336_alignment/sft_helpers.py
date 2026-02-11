# sft_helpers.py
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams
from typing import Dict, Callable
import json
from tqdm import tqdm
from xopen import xopen
from unittest.mock import patch
from vllm.model_executor import set_random_seed as vllm_set_random_seed
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from datasets import load_dataset

def to_r1_zero(example):
    current_dir = Path(__file__).resolve().parent
    with open(f"{current_dir}/prompts/r1_zero.prompt", "r") as f:
        prompt_prefix = f.read()
    q = example["question"].strip()
    a = example["answer"].strip()
    if "\n####" in a:
        reasoning, final = a.rsplit("\n####", 1) # split on last \n####
        reasoning = reasoning.strip()
        final = final.strip()
    else:
        reasoning = ""
        final = a
    prompt = prompt_prefix.format(question=q)
    # build the exact model completion as a separate field
    completion = "{reasoning} </think> <answer> {final} </answer>".format(reasoning=reasoning, final=final)
    return {
        "prompt": prompt,
        "reasoning": reasoning,
        "answer": final,
        "completion": completion, 
    }
    
def load_gsm8k_dataset(max_train_samples, max_eval_samples, seed):
    parent_dir = Path(__file__).resolve().parent.parent
    ds = load_dataset(f"{parent_dir}/data/gsm8k")
    ds = ds.map(to_r1_zero, remove_columns=ds["train"].column_names)
    if max_train_samples is not None:
        train_ds = ds["train"].shuffle(seed=seed).select(range(max_train_samples))
        print(train_ds)
    else:
        train_ds = ds["train"]
    if max_eval_samples is not None:
        valid_ds = ds["test"].shuffle(seed=seed).select(range(max_eval_samples))
    else:
        valid_ds = ds["test"]
    return train_ds, valid_ds

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
    """
    Tokenize the prompt and output strings, and construct a mask that is 1 for the 
    response tokens and 0 for other tokens (prompt or padding).
    """
    prompt_lens = []
    full_text_tokens = []
    for prompt_str, output_str in zip(prompt_strs, output_strs):
        p = tokenizer.encode(prompt_str)
        o = tokenizer.encode(output_str)
        prompt_lens.append(len(p))
        full_text_tokens.append(p + o)
    max_length = max([len(f) for f in full_text_tokens])

    input_ids = torch.full((len(prompt_strs), max_length - 1), tokenizer.pad_token_id)
    labels = torch.full((len(prompt_strs), max_length - 1), tokenizer.pad_token_id)
    response_mask = torch.full((len(prompt_strs), max_length - 1), False)
    for i, (p_len, f) in enumerate(zip(prompt_lens, full_text_tokens)):
        if len(f) < max_length:
            input_ids[i, :len(f)] = torch.tensor(f)
        else:
            input_ids[i, :len(f) - 1] = torch.tensor(f[:-1])
        labels[i, :len(f) - 1] = torch.tensor(f[1:])
        response_mask[i, p_len - 1:len(f) - 1] = True
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).
    Args:
        logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size)
        containing unnormalized logits.
    Returns:
        torch.Tensor Shape (batch_size, sequence_length): the entropy of the next-token predictions.
    """
    logits_f = logits.float()
    probs = F.softmax(logits_f, dim=-1)
    log_probs = F.log_softmax(logits_f, dim=-1)
    return -torch.sum(probs * log_probs, dim=-1)


def get_response_log_probs(model, input_ids, labels, return_token_entropy=False):
    logits = model(input_ids).logits              # bf16 forward OK
    logits_f = logits.float()                     # ✅ critical
    log_probs = F.log_softmax(logits_f, dim=-1)    # fp32
    # if labels are pad_token_id, gather is OK; you'll mask later
    response_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # fp32
    if return_token_entropy:
        with torch.no_grad():
            token_entropy = compute_entropy(logits_f)   # ✅ entropy in fp32
        return {"log_probs": response_log_probs, "token_entropy": token_entropy}
    return {"log_probs": response_log_probs}


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float = 1.0,
    dim: int | None = None,
) -> torch.Tensor:

    """ Sum over a dimension and normalize by a constant, considering only those elements where mask
    == 1.
    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, same shape as tensor; positions with 1 are included in the sum.
        normalize_constant: float, the constant to divide by for normalization.
        dim: int | None, the dimension to sum along before normalization. If None, sum over all
            dimensions.
    Returns:
        torch.Tensor, the normalized sum, where masked elements (mask == 0) don't contribute to
            the sum.
    """
    return torch.sum(tensor * mask, dim=dim) / normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Execute a forward-and-backward pass on a microbatch.
    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            per-token log-probabilities from the SFT policy being trained.
        response_mask: torch.Tensor of shape (batch_size, sequence_length):
            1 for response tokens, 0 for prompt/padding.
        gradient_accumulation_steps: int, the number of microbatches per optimizer step.
        normalize_constant: float, the constant by which to divide the sum. It is fine to leave this as 1.0.
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            loss: scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
                this so we can log it.
            metadata: Dict with metadata from the underlying loss call, and any other statistics you
                might want to log.
    """
    # Negative log probability loss (we want to maximize log prob, so minimize -log prob)
    # Divide by gradient_accumulation_steps to properly scale gradients across microbatches
    loss = -masked_normalize(policy_log_probs, response_mask, normalize_constant, dim=-1).mean()
    loss = loss / gradient_accumulation_steps
    loss.backward()
    metadata = {
        "loss": loss.item(),
    }
    return loss, metadata

def log_generations(
    input_prompts: list[str],
    ground_truth_answers: list[str],
    llm: LLM,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_sampling_params: SamplingParams,
    reward_fn: Callable,
    output_path: str,
    batch_size: int,
    include_token_entropy: bool = False,
):
    """Log the generations of the model.
    Log the following information for each example:
        - input_prompt: str, the input prompt.
        - model_response: str, the response generated by the model.
        - ground_truth_answer: str, the ground truth answer.
        - reward_info: Dict[str, float], the reward information.
        - average_token_entropy: float, the average token entropy.
        - average_response_length: float, the average response length
    """
    model_responses = llm.generate(input_prompts, eval_sampling_params)
    if include_token_entropy:
        tokenized_data = tokenize_prompt_and_output(list(input_prompts), [model_response.outputs[0].text for model_response in model_responses], tokenizer)
        input_ids, response_mask = tokenized_data["input_ids"], tokenized_data["response_mask"]
        # compute average token entropy and response log probability
        model.eval()
        with torch.no_grad():
            # fill with infs
            average_token_entropy_list = torch.ones(len(input_prompts)) * float('inf')
            for i in np.arange(0, len(input_prompts), batch_size):
                logits = model(input_ids[i:i+batch_size].to(model.device)).logits
                token_entropy = compute_entropy(logits).cpu()
                # average token entropy per sequence
                average_token_entropy_list[i:i+batch_size] = (token_entropy * response_mask[i:i+batch_size]).sum(dim=-1) / response_mask[i:i+batch_size].sum(dim=-1)
        model.train()
    else:
        average_token_entropy_list = len(input_prompts) * [float('inf')]
    log_infos = []
    for input_prompt, ground_truth_answer, model_response, average_token_entropy in \
        tqdm(zip(input_prompts, ground_truth_answers, model_responses, average_token_entropy_list)):
        reward_info = reward_fn(model_response.outputs[0].text, ground_truth_answer)
        log_infos.append({
            "input_prompt": input_prompt,
            "model_response": model_response.outputs[0].text,
            "ground_truth_answer": ground_truth_answer,
            "format_reward": reward_info["format_reward"],
            "answer_reward": reward_info["answer_reward"],
            "reward": reward_info["reward"],
            "response_length": len(model_response.outputs[0].text.split()),
            "average_token_entropy": float(average_token_entropy),
        })
        
    stats = {
        "count_format_0": sum(1 for log_info in log_infos if log_info["format_reward"] == 0),
        "count_format_1_answer_0": sum(1 for log_info in log_infos if log_info["format_reward"] == 1 and log_info["answer_reward"] == 0),
        "count_format_1_answer_1": sum(1 for log_info in log_infos if log_info["format_reward"] == 1 and log_info["answer_reward"] == 1),
        "average_format_reward": float(np.mean([log_info["format_reward"] for log_info in log_infos])),
        "average_answer_reward": float(np.mean([log_info["answer_reward"] for log_info in log_infos])),
        "average_reward": float(np.mean([log_info["reward"] for log_info in log_infos])),
        "average_response_length": float(np.mean([log_info["response_length"] for log_info in log_infos])),
        "average_response_length_correct": float(np.mean([log_info["response_length"] for log_info in log_infos if log_info["reward"] == 1])),
        "average_response_length_incorrect": float(np.mean([log_info["response_length"] for log_info in log_infos if log_info["reward"] == 0])),
        "average_token_entropy": float(np.mean([log_info["average_token_entropy"] for log_info in log_infos])),
    }
    with xopen(output_path, "w") as f:
        f.write(json.dumps(stats) + "\n")
        for log_info in log_infos:
            f.write(json.dumps(log_info) + "\n")
    return stats


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    Args:
        model_id: str, the path to the model.
        device: str, the device to use for the model.
        seed: int, the seed to use for the model.
        gpu_memory_utilization: float, the memory utilization to use for the model.
    Returns:
        LLM, the vLLM model.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        llm =  LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        # clean up the cache
        torch.cuda.empty_cache()
        return llm

def load_policy_into_vllm_instance(
    policy: PreTrainedModel, 
    llm: LLM,
):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def gradient_clipping(
    policy: PreTrainedModel,
    max_grad_norm: float,
):
    """
    Clip the gradients of the policy.
    """
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)

def cuda_mem_snapshot(device: str | int = 0):
    dev = torch.device(device) if isinstance(device, str) else torch.device(f"cuda:{device}")

    torch.cuda.synchronize(dev)
    allocated = torch.cuda.memory_allocated(dev)
    reserved  = torch.cuda.memory_reserved(dev)
    free, total = torch.cuda.mem_get_info(dev)  # bytes

    print(f"CUDA memory snapshot on {dev}:")
    print(f"  allocated: {allocated / 1024**3:.2f} GiB")
    print(f"  reserved : {reserved  / 1024**3:.2f} GiB")
    print(f"  free     : {free      / 1024**3:.2f} GiB")
    print(f"  total    : {total     / 1024**3:.2f} GiB")

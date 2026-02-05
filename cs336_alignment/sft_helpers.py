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

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1 for the 
    response tokens and 0 for other tokens (prompt or padding).
    
    Args:
        prompt_strs: List of prompt strings.
        output_strs: List of output strings.
        tokenizer: Tokenizer to use for tokenization.
        
    Returns:
        Dictionary containing the tokenized prompt and output strings, and a mask that 
        is 1 for the response tokens and 0 for other tokens (prompt or padding).
    """
    prompt_tokens = tokenizer(prompt_strs, return_tensors="pt", padding=True, truncation=False)
    output_tokens = tokenizer(output_strs, return_tensors="pt", padding=True, truncation=False)
    
    # Calculate max length needed (sum of actual tokens, not padding)
    max_length = max(
        p_mask.sum().item() + o_mask.sum().item() 
        for p_mask, o_mask in zip(prompt_tokens['attention_mask'], output_tokens['attention_mask'])
    )
    
    tokenized_data = {
        "input_ids": [],
        "labels": [],
        "response_mask": [],
    }
    
    for p_ids, p_mask, o_ids, o_mask in zip(
        prompt_tokens['input_ids'], 
        prompt_tokens['attention_mask'], 
        output_tokens['input_ids'], 
        output_tokens['attention_mask']
    ):
        # Extract actual tokens (without padding) and convert to list
        p_ids_list = p_ids[p_mask.bool()]
        o_ids_list = o_ids[o_mask.bool()]
        # Concatenate prompt and output
        full_ids = torch.cat([p_ids_list, o_ids_list])
        # Initialize with padding tokens
        input_ids = full_ids.new_full((max_length - 1, ), tokenizer.pad_token_id)
        labels = full_ids.new_full((max_length - 1, ), tokenizer.pad_token_id)
        response_mask = full_ids.new_full((max_length - 1, ), False)
        # Copy prompt and output tokens (shifted for next-token prediction)
        if len(full_ids) < max_length:
            input_ids[:len(full_ids)] = full_ids
        else:
            input_ids[:len(full_ids) - 1] = full_ids[:-1]
        labels[:len(full_ids) - 1] = full_ids[1:]
        # Set response_mask to 1 for output tokens (shifted by 1)
        response_mask[len(p_ids_list) - 1:len(full_ids) - 1] = True
        tokenized_data["input_ids"].append(input_ids)
        tokenized_data["labels"].append(labels)
        tokenized_data["response_mask"].append(response_mask)

    return {k: torch.stack(v) for k, v in tokenized_data.items()}

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).
    Args:
        logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size)
        containing unnormalized logits.
    Returns:
        torch.Tensor Shape (batch_size, sequence_length): the entropy of the next-token predictions.
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -torch.sum(probs * log_probs, dim=-1)

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Get the conditional log-probs of the response given the prompt,
    and optionally the entropy of the next token predictions.
    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.
    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions.
    """
    logits = model(input_ids).logits
    #logits = logits - torch.max(logits, dim=-1, keepdim=True).values
    #log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    log_probs = F.log_softmax(logits, dim=-1)
    response_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    if return_token_entropy:
        return {
            "log_probs": response_log_probs,
            "token_entropy": compute_entropy(logits),
        }
    else:
        return {
            "log_probs": response_log_probs,
        }

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
        - average_response_log_prob: float, the average response log probability.
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
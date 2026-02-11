# grpo_helpers.py
import torch
from typing import Callable, Literal
from einops import rearrange
from vllm import LLM, SamplingParams
from .sft_helpers import masked_normalize

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """ Compute rewards for each group of rollout responses, normalized by the group size. """
    rollout_batch_size = len(rollout_responses)
    metadata = {
        "format_reward": torch.zeros(rollout_batch_size),
        "answer_reward": torch.zeros(rollout_batch_size),
    }
    raw_rewards = torch.zeros(rollout_batch_size)
    for i, (rollout_response, repeated_ground_truth) in enumerate(zip(rollout_responses, repeated_ground_truths)):
        reward = reward_fn(rollout_response, repeated_ground_truth)
        raw_rewards[i] = reward["reward"]
        metadata["format_reward"][i] = reward["format_reward"]
        metadata["answer_reward"][i] = reward["answer_reward"]
    # reshape raw rewards to (n_prompts_per_rollout_batch, group_size)
    raw_rewards = rearrange(raw_rewards, "(n_prompts gs) -> n_prompts gs", gs=group_size)
    mean_rewards = raw_rewards.mean(dim=-1, keepdim=True)
    if normalize_by_std:
        advantage = (raw_rewards - mean_rewards) / (raw_rewards.std(dim=-1, keepdim=True) + advantage_eps)
    else:
        advantage = (raw_rewards - mean_rewards)
    # reshape raw_rewards, advantage to (rollout_batch_size,)
    raw_rewards = rearrange(raw_rewards, "n_prompts gs -> (n_prompts gs)")
    advantage = rearrange(advantage, "n_prompts gs -> (n_prompts gs)")
    return advantage, raw_rewards, metadata

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """ Compute the naive policy gradient loss. """
    return -raw_rewards_or_advantages.float() * policy_log_probs.float()

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """ Compute the GRPO-Clip loss. """
    # fp32 everywhere for the ratio path
    adv = advantages.float()
    weights = torch.exp((policy_log_probs.float() - old_log_probs.float()).clamp(-10, 10))
    unclipped_loss = weights * adv
    clipped_loss = torch.clamp(weights, 1 - cliprange, 1 + cliprange) * adv
    loss = -torch.min(unclipped_loss, clipped_loss)
    clipped_mask = (weights > 1 + cliprange) | (weights < 1 - cliprange)
    metadata = {"clip_fraction": clipped_mask.float().mean().item()}
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """ Select and compute the desired policy-gradient loss. """
    metadata = {"loss_type": loss_type}
    if loss_type == "no_baseline":
        assert raw_rewards is not None
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
    elif loss_type == "grpo_clip":
        assert advantages is not None
        assert old_log_probs is not None
        assert cliprange is not None
        loss, metadata_grpo_clip = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
        metadata.update(metadata_grpo_clip)
    else:   
        raise ValueError(f"Invalid loss type: {loss_type}")
    return loss, metadata

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None= None,
) -> torch.Tensor:
    """ Compute the mean of tensor along a given dimension, considering only those elements where mask == 1. """
    return torch.sum(tensor * mask, dim=dim) / torch.sum(mask, dim=dim).clamp(min=1)

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
    normalize_by_length: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """ Execute a forward-and-backward pass on a microbatch. """
    loss, metadata = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
    if normalize_by_length:
        loss = masked_normalize(
            loss,
            response_mask, 
            normalize_constant=float(response_mask.shape[-1]), 
            dim=-1,
        ).mean() / gradient_accumulation_steps
    else:
        loss = masked_mean(loss, response_mask) / gradient_accumulation_steps
    loss.backward()
    metadata["loss_for_logging"] = loss.item() # Not the right evaluation metric !!!
    metadata["average_response_length"] = response_mask.sum(dim=-1).float().mean().item()
    return loss, metadata

def shuffle_groups(
    rollout_batch_size: int,
    group_size: int,
) -> torch.Tensor:
    """ Shuffle the rollout batch. """
    shuffle_idx = torch.arange(rollout_batch_size)
    shuffle_idx = rearrange(shuffle_idx, "(n_prompts gs) -> n_prompts gs", gs=group_size)
    shuffle_idx = shuffle_idx[torch.randperm(shuffle_idx.shape[0])]
    shuffle_idx = rearrange(shuffle_idx, "n_prompts gs -> (n_prompts gs)")
    return shuffle_idx
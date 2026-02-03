import torch
from transformers import PreTrainedTokenizer, PreTrainedModel

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
        print(p_ids)
        print(p_ids_list)
        print(o_ids)
        print(o_ids_list)
        
        # Concatenate prompt and output
        full_ids = torch.cat([p_ids_list, o_ids_list])
        print(full_ids)
        
        # Initialize with padding tokens
        input_ids = full_ids.new_full((max_length - 1, ), tokenizer.pad_token_id)
        labels = full_ids.new_full((max_length - 1, ), tokenizer.pad_token_id)
        response_mask = full_ids.new_full((max_length - 1, ), False)
        
        # Copy prompt and output tokens (shifted for next-token prediction)
        #input_ids[:len(full_ids) - 1] = torch.tensor(full_ids[:-1])
        print(max_length, len(full_ids))
        if len(full_ids) < max_length:
            input_ids[:len(full_ids)] = full_ids
        else:
            input_ids[:len(full_ids) - 1] = full_ids[:-1]
        labels[:len(full_ids) - 1] = full_ids[1:]
        
        # Set response_mask to 1 for output tokens (shifted by 1)
        response_mask[len(p_ids_list) - 1:len(full_ids) - 1] = True
        print("input_ids")
        print(input_ids[:10])
        print("labels")
        print(labels[:10])
        print("response_mask")
        print(response_mask[:10])
        
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
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return -torch.sum(torch.exp(log_probs) * log_probs, dim=-1)

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
    outputs = model(input_ids)
    print(outputs)
    logits = model(input_ids).logits
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    response_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        return {
            "log_probs": response_log_probs,
            "token_entropy": token_entropy,
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

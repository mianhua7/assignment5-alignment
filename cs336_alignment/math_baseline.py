from datasets import load_dataset
from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from typing import Callable, List
import json
import os
import regex as re
from tqdm import tqdm 
from pathlib import Path


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
    completion = "{reasoning}</think> <answer>{final}</answer>".format(reasoning=reasoning, final=final)
    return {
        "prompt": prompt,
        "reasoning": reasoning,
        "answer": final,
        "completion": completion, 
    }

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers: List[str],
    eval_sampling_params: SamplingParams,
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute the reward for each prompt, and return the average reward.
    """
    print(f"Generating {len(prompts)} completions")
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    results = []
    print(f"Evaluating {len(prompts)} prompts")
    for i, output in tqdm(enumerate(outputs)):
        model_completion = output.outputs[0].text
        reward = reward_fn(model_completion, answers[i])
        if not isinstance(reward, dict):
            reward = {"reward": reward}
        results.append({
            "prompt": prompts[i],
            "answer": answers[i],
            "model_completion": model_completion,
            **reward,
        })
    return results

if __name__ == "__main__":
    # load dataset, ../data/gsm8k/test.jsonl
    parent_dir = Path(__file__).resolve().parent.parent
    dataset = load_dataset("json", data_files={"test": f"{parent_dir}/data/gsm8k/test.jsonl"})

    # format dataset as string prompts to the language model using the r1_zero prompt format
    formatted_dataset = dataset.map(to_r1_zero, remove_columns=dataset["test"].column_names)

    prompts = formatted_dataset["test"]["prompt"]
    answers = formatted_dataset["test"]["answer"]
    completions = formatted_dataset["test"]["completion"]
    # sanity check that the completions are correct
    rewards = r1_zero_reward_fn(completions[0], answers[0])
    print(rewards)
    if rewards["reward"] == 0:
        print("Sanity check failed")
        exit(1)

    # load model from local path
    parent_dir = Path(__file__).resolve().parent.parent
    model_path = f"{parent_dir}/models/Qwen2.5-Math-1.5B"
    vllm_model = LLM(model=model_path, trust_remote_code=True)
    
    # load eval sampling params
    eval_sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024)
    # evaluate vllm
    results= evaluate_vllm(vllm_model, r1_zero_reward_fn, prompts, answers, eval_sampling_params)
    # serialize results to disk
    output_dir = f"{parent_dir}/exps/math_baseline"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/gsm8k_results.jsonl"
    print(f"Saving results to {output_path}")
    with open(output_path, "w") as f:
        for result in tqdm(results):
            f.write(json.dumps(result) + "\n")
    print(f"Done saving results to {output_path}")
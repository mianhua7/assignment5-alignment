# Zero-shot evaluation of Llama 3.1 8B on MMLU.
# (1) Load MMLU examples, (2) format as prompts with system + MMLU prompt,
# (3) generate with stop at "# Query:", (4) compute metrics, (5) serialize to disk.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

from cs336_alignment.utils import (
    load_mmlu_dataset,
    format_mmlu_example,
    parse_mmlu_response,
)


def evaluate_llm(
    llm: LLM,
    prompts: list[str],
    answers: list[str],
    sampling_params: SamplingParams,
) -> list[dict]:
    """Generate completions, parse answers, and return per-example results."""
    print(f"Generating {len(prompts)} completions")
    outputs = llm.generate(prompts, sampling_params)
    results = []
    for i, output in tqdm(enumerate(outputs), total=len(outputs), desc="Evaluating"):
        model_completion = output.outputs[0].text
        parsed = parse_mmlu_response(model_completion)
        format_reward = 1.0 if parsed is not None else 0.0
        answer_reward= 1.0 if parsed == answers[i] else 0.0
        results.append({
            "model_completion": model_completion,
            "parsed_answer": parsed,
            "gold_answer": answers[i],
            "format_reward": format_reward,
            "answer_reward": answer_reward,
            "reward": format_reward * answer_reward,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Zero-shot MMLU evaluation with Llama 3.1 8B")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test", "val"])
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default=f"{Path(__file__).resolve().parent.parent}/exps/mmlu_baseline")
    
    args = parser.parse_args()
    # Load dataset and format prompts
    ds = load_mmlu_dataset(args.split)
    ds = ds.map(format_mmlu_example, remove_columns=ds.column_names, desc="Format prompts")
    prompts = ds["prompt"]
    answers = ds["answer"]
    ds_full = load_mmlu_dataset(args.split)

    print("Finished loading and formatting dataset")

    # Load model and initialize sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        stop=["# Query:"],
        include_stop_str_in_output=False,
    )
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        seed=42,
        tokenizer_mode="slow",
    )
    print("Finished loading model")

    # Generate and evaluate completions
    results = evaluate_llm(llm, prompts, answers, sampling_params)
    print("Finished generating outputs")
    
    # Compute metrics
    n = len(results)
    average_format_reward   = np.mean([r["format_reward"] for r in results])
    average_answer_reward = np.mean([r["answer_reward"] for r in results])
    average_reward = np.mean([r["reward"] for r in results])

    print("Metrics:")
    print(f"  Total examples: {n}")
    print(f"  Average format reward: {average_format_reward:.4f}")
    print(f"  Average answer reward: {average_answer_reward:.4f}")
    print(f"  Average reward: {average_reward:.4f}")

    # Serialize examples, generations, and scores to disk
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"mmlu_baseline_results_{args.split}.jsonl"
    metrics_path = output_dir / f"mmlu_baseline_metrics_{args.split}.json"

    records = []
    for i, r in enumerate(results):
        ex = ds_full[i]
        records.append({
            "subject": ex["subject"],
            "question": ex["question"],
            "options": ex["options"],
            "gold_answer": ex["answer"],
            "prompt": prompts[i],
            "model_completion": r["model_completion"],
            "parsed_answer": r["parsed_answer"],
            "format_reward": r["format_reward"],
            "answer_reward": r["answer_reward"],
            "reward": r["reward"],
        })

    with open(results_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Results written to {results_path}")

    metrics = {
        "split": args.split,
        "model": args.model,
        "n": n,
        "average_format_reward": float(average_format_reward),
        "average_answer_reward": float(average_answer_reward),
        "average_reward": float(average_reward),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()

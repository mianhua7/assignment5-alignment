from datetime import datetime
from pathlib import Path
from datasets import load_dataset
import random
import os
import numpy as np
from pathlib import Path

# read results from exps/math_baseline/gsm8k_results.jsonl
ds = load_dataset("json", data_files={"results": "exps/math_baseline/gsm8k_results.jsonl"})["results"]
out_path = Path(__file__).parent.parent / "report" / "report.md"
out_path.parent.mkdir(parents=True, exist_ok=True)

lines = []
lines.append(f"# CS336 Assignment 5 Report\n")

lines.append("## Summary\n")
lines.append("- Model: Qwen/Qwen2.5-Math-1.5B\n")
lines.append("- Dataset: gsm8k test\n\n")

lines.append("## Metrics\n")
lines.append("| metric | value |\n|---|---:|\n")
lines.append(f"| format_reward | {np.mean(ds["format_reward"])} |\n")
lines.append(f"| answer_reward | {np.mean(ds["answer_reward"])} |\n")
lines.append(f"| reward | {np.mean(ds["reward"])} |\n\n")

lines.append("## Ratio of each category\n")
lines.append("| metric | ratio |\n|---|---:|\n")
# Convert columns to numpy arrays for element-wise operations
format_rewards = np.array(ds["format_reward"])
answer_rewards = np.array(ds["answer_reward"])
lines.append(f"| correct_with_both_reward_1 | {np.mean(format_rewards * answer_rewards)} |\n")
lines.append(f"| format_reward_1_answer_reward_0 | {np.mean(format_rewards * (1 - answer_rewards))} |\n")
lines.append(f"| format_reward_0_answer_reward_0 | {np.mean((1 - format_rewards) * (1 - answer_rewards))} |\n\n")

# show 10 examples with format_reward = 0
lines.append("## Examples with format_reward = 0\n")
lines.append("| prompt | model_completion |\n|---|---|\n")
ds_filtered = ds.filter(lambda x: x["format_reward"] == 0)
for result in random.sample(list(ds_filtered), min(10, len(ds_filtered))):
    lines.append(f"| {result['prompt']} | {result['model_completion']} |\n")

out_path.write_text("".join(lines), encoding="utf-8")
print(f"Report saved to {out_path}")

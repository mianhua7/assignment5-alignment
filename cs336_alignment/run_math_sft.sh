#!/usr/bin/env bash
# run math_sft.py with varying the number of unique examples for SFT in the range {128, 256, 512, 1024, None}, along with using the full dataset. 
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exp_dir="${script_dir}/../exps/math_sft"

export CUDA_VISIBLE_DEVICES=0,1

for num_samples in 128 256 512 1024; do
    uv run python math_sft.py --max-train-samples $num_samples --output-dir $exp_dir/${num_samples}_samples --max-epochs 20
done

uv run python math_sft.py --output-dir $exp_dir/all_samples --max-epochs 10
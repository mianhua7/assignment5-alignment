#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1
# parent director of the script
parent_dir="$(dirname $(dirname $(realpath $0)))"
exp_dir="${parent_dir}/exps/ei_math"
for num_rollouts in 1 2 4 8; do 
    for n_sft_epochs in 1 2 4; do 
        uv run python expert_iteration.py --num-rollouts $num_rollouts --n-sft-epochs $n_sft_epochs --ei-batch-size 512 --output-dir $exp_dir/${num_rollouts}_rollouts_${n_sft_epochs}_sft_epochs_512_ei_batch
    done
done
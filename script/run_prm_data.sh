#!/bin/bash
# Generate PRM training data from GraphInstruct-RFT-Aug traces.
#
# Parses each reasoning trace into steps, labels correct steps as positive,
# generates corrupted versions as negative examples.
#
# Output: dataset/PRM/train.jsonl
#
# Usage:
#   bash script/run_prm_data.sh
#   bash script/run_prm_data.sh --max_samples 10000  # quick test

set -e

python training/prm/generate_prm_data.py \
    --input dataset/GraphInstruct-RFT-Aug/train.jsonl \
    --output_dir dataset/PRM \
    --neg_ratio 1.0 \
    --val_frac 0.05 \
    --seed 42 \
    "$@"
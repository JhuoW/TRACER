#!/bin/bash
# sft_v4: RFT-Aug dataset + full-sequence loss + higher batch diversity
#
# Key changes vs sft_v3:
#   1. Dataset: GraphInstruct-RFT-Aug (RFT diverse reasoning + permutation aug)
#      - 71,705 RFT samples × 3 (1 anchor + k=2 perms) = ~215K samples
#      - Much better class balance (e.g., cycle: 91% Yes vs 98.8% before)
#      - 2x more unique graphs, multiple reasoning paths per graph
#   2. Loss: full-sequence (--no_mask_prompt) — loss on prompt + response
#   3. Batch diversity via gradient accumulation:
#      groups_per_batch=1 (TRA NaN constraint), but accum=10 compensates:
#      1 graph/micro × 3 GPUs × 10 accum = 30 unique graphs/update (≈ GraphWiz's 32)
#      Total opt steps: ~4,780 (2 epochs), close to GraphWiz's ~4,480.
#   4. k=2 permutations (reduced from 4 — RFT already provides diversity)
#
#   5. No [GRAPH_REPR] token — eliminates train/eval mismatch.
#      L_repr uses last prompt token instead.  Dataset generated with --no_graph_repr.
#
# Generate the dataset first:
#   python dataset/generate_permuted.py rft --k 2 --seed 42 --no_graph_repr
#
# Usage:
#   bash script/run_sft.sh [OUTPUT_DIR] [GPUS]
#
# Evaluate (no merge needed — full model):
#   bash script/run_eval.sh checkpoints/sft cycle vllm 1
#   bash script/run_eval.sh checkpoints/sft all hf-para "0,1,2"
set -e

OUTPUT_DIR=${1:-"checkpoints/sft"}
GPUS=${2:-"0,1,2"}

# Kill stale processes on the master port
kill $(lsof -ti:29501) 2>/dev/null || true
sleep 1

deepspeed --include "localhost:${GPUS}" \
    --master_port 29501 \
    training/sft_train.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --data_path dataset/GraphInstruct-RFT-Aug/train.jsonl \
    --attn_implementation flash_attention_2 \
    --r 8 \
    --d_max_fp 10 \
    --lambda_repr 0.1 \
    --k 2 \
    --groups_per_batch 1 \
    --no_mask_prompt \
    --max_seq_len 2048 \
    --gradient_accumulation_steps 10 \
    --num_train_epochs 2 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --warmup_steps 200 \
    --weight_decay 0.0 \
    --gradient_checkpointing \
    --zero_stage 3 \
    --seed 42 \
    --logging_steps 10 \
    --save_per_epoch \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project llm-gc-sft \
    --wandb_run_name "sft-v4-rft-aug"

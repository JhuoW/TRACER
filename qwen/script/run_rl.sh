#!/bin/bash
# GRPO (Phase 2) for the Qwen2.5 backbone variant.
#
# Reuses training/rl_train.py unchanged — RL does not touch TRA, so the
# original trainer is backbone-agnostic. We only override the input /
# output paths so the Qwen pipeline stays isolated from the LLaMA one.
#
# Usage:
#   bash qwen/script/run_rl.sh [CHECKPOINT_DIR] [OUTPUT_DIR] [GPUS]
#
# Examples:
#   bash qwen/script/run_rl.sh
#   bash qwen/script/run_rl.sh \
#       checkpoints/Qwen2.5-3B-Instruct/sft \
#       checkpoints/Qwen2.5-3B-Instruct/rl "0,1,2"

set -e

CHECKPOINT_DIR=${1:-"checkpoints/Qwen2.5-3B-Instruct/sft"}
OUTPUT_DIR=${2:-"checkpoints/Qwen2.5-3B-Instruct/rl"}
GPUS=${3:-"0,1,2"}

kill $(lsof -ti:29512) 2>/dev/null || true
sleep 1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "  LLM-GC GRPO Training (Qwen backbone)"
echo "=========================================="
echo "  Base ckpt : ${CHECKPOINT_DIR}"
echo "  Output    : ${OUTPUT_DIR}"
echo "  GPUs      : ${GPUS}"
echo "=========================================="

deepspeed --include "localhost:${GPUS}" \
    --master_port 29512 \
    training/rl_train.py \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --data_path dataset/GraphInstruct-RFT-Aug/train.jsonl \
    --max_samples 10000 \
    --reward_type shaped \
    --num_generations 8 \
    --num_iterations 1 \
    --epsilon 0.2 \
    --temperature 1.0 \
    --top_p 0.95 \
    --max_new_tokens 1024 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1 \
    --learning_rate 5e-7 \
    --warmup_steps 50 \
    --gradient_checkpointing \
    --seed 42 \
    --logging_steps 10 \
    --save_steps 500 \
    --output_dir "${OUTPUT_DIR}" \
    --wandb_project llm-gc-rl \
    --wandb_run_name "grpo-qwen-v1"

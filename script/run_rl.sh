#!/bin/bash
# Phase 2: GRPO training starting from SFT checkpoint.
#
# Reinforces correct graph reasoning via outcome-based rewards.
# Uses DeepSpeed ZeRO-2 (full model on each GPU for generation).
#
# Memory budget (8B model, ZeRO-2, 3 GPUs):
#   Model bf16:       16 GB (replicated)
#   Optimizer states: ~32 GB (sharded → ~11 GB/GPU)
#   Gradients:        ~16 GB (sharded → ~5 GB/GPU)
#   Static total:     ~32 GB/GPU
#   → batch_size=1, G=8 fits on 80+ GB GPUs
#
# Key hyperparameters:
#   temperature=1.0  — high diversity so G completions differ (GRPO needs variance)
#   reward=shaped    — format bonus + numeric partial credit (more signal than binary)
#   max_samples=10K  — manageable training time (~8h on 3 GPUs)
#   lr=5e-7          — conservative RL learning rate (10x lower than SFT)
#
# Usage:
#   bash script/run_rl.sh [CHECKPOINT_DIR] [OUTPUT_DIR] [GPUS]
#
# Examples:
#   bash script/run_rl.sh                                          # defaults
#   bash script/run_rl.sh checkpoints/sft checkpoints/rl "0,1,2"
#
# Evaluate:
#   bash script/run_eval.sh checkpoints/rl cycle hf 0
#   bash script/run_eval.sh --vllm checkpoints/rl all

set -e

CHECKPOINT_DIR=${1:-"checkpoints/sft"}
OUTPUT_DIR=${2:-"checkpoints/rl"}
GPUS=${3:-"0,1,2"}

# Kill stale processes on the master port
kill $(lsof -ti:29502) 2>/dev/null || true
sleep 1

# Reduce CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "  LLM-GC GRPO Training"
echo "=========================================="
echo "  SFT checkpoint : ${CHECKPOINT_DIR}"
echo "  Output         : ${OUTPUT_DIR}"
echo "  GPUs           : ${GPUS}"
echo "=========================================="

deepspeed --include "localhost:${GPUS}" \
    --master_port 29502 \
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
    --wandb_run_name "grpo-v1"

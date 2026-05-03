#!/bin/bash
# Train Process Reward Model (PRM) on step-level correctness labels.
#
# Fine-tunes Qwen-2.5-7B with LoRA + binary classification head.
# Uses DeepSpeed ZeRO-2 for multi-GPU training.
#
# Prerequisites:
#   1. Generate PRM data first: bash script/run_prm_data.sh
#   2. Ensure Qwen-2.5-7B is accessible (auto-downloads from HuggingFace)
#
# Memory budget (7B model, LoRA, ZeRO-2, 3 GPUs):
#   Model bf16:       ~14 GB (replicated, but only LoRA grads)
#   LoRA params:      ~50 MB
#   Optimizer states: ~100 MB (sharded, LoRA only)
#   → batch_size=4 fits comfortably on 80 GB GPUs
#
# Usage:
#   bash script/run_prm_train.sh [GPUS]
#
# Examples:
#   bash script/run_prm_train.sh                  # defaults (GPUs 0,1,2)
#   bash script/run_prm_train.sh "0,1"            # 2 GPUs

set -e

GPUS=${1:-"0,1,2"}

# Kill stale processes on the master port
kill $(lsof -ti:29503) 2>/dev/null || true
sleep 1

# Reduce CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "  LLM-GC PRM Training (Qwen-2.5-7B)"
echo "=========================================="
echo "  GPUs           : ${GPUS}"
echo "=========================================="

deepspeed --include "localhost:${GPUS}" \
    --master_port 29503 \
    training/prm/prm_train.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --data_path dataset/PRM/train.jsonl \
    --val_data_path dataset/PRM/val.jsonl \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 2 \
    --learning_rate 1e-4 \
    --head_lr_mult 10.0 \
    --lr_scheduler_type cosine \
    --warmup_steps 100 \
    --gradient_checkpointing \
    --seed 42 \
    --logging_steps 10 \
    --save_steps 1000 \
    --eval_steps 500 \
    --output_dir checkpoints/prm \
    --wandb_project llm-gc-prm \
    --wandb_run_name "prm-qwen-7b"
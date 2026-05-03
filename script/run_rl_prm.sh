#!/bin/bash
# Phase 3: GRPO with compound reward (shaped + PRM trace score).
#
# Continues training from the current RL checkpoint. The PRM is the trained
# process reward model from step 6 — it scores each intermediate reasoning
# step, and those scores are aggregated into a trace-level reward:
#
#     R = (1 - w_trace) * R_shaped + w_trace * R_PRM
#
# Shaped correctness remains the dominant signal (avoids reward hacking);
# the PRM just encourages step-level consistency.
#
# Memory budget (8B RL + 7B PRM, bf16, 3 GPUs):
#   RL model    :  ~16 GB replicated + ZeRO-2 sharded grads/optim
#   PRM (Qwen-7B): ~14 GB replicated on each rank (frozen, eval only)
#   Static      :  ~40-45 GB/GPU — comfortable on 80+ GB cards
#
# Throughput:
#   Each GRPO step adds one PRM batched forward per rollout group.
#   Expected slowdown vs run_rl.sh: roughly 1.5-2x.
#
# Usage:
#   bash script/run_rl_prm.sh [RL_CKPT] [PRM_CKPT] [OUTPUT_DIR] [GPUS]
#
# Examples:
#   bash script/run_rl_prm.sh
#   bash script/run_rl_prm.sh checkpoints/rl checkpoints/prm/best checkpoints/rl_prm "0,1,2"
#
# Evaluate:
#   bash script/run_eval.sh --vllm checkpoints/rl_prm all

set -e

RL_CKPT=${1:-"checkpoints/rl"}
PRM_CKPT=${2:-"checkpoints/prm/best"}
OUTPUT_DIR=${3:-"checkpoints/rl_prm"}
GPUS=${4:-"0,1,2"}

# Fail early if the PRM checkpoint is missing.
if [ ! -f "${PRM_CKPT}/prm_config.json" ]; then
    echo "ERROR: PRM checkpoint not found at ${PRM_CKPT}"
    echo "       (expected ${PRM_CKPT}/prm_config.json)"
    exit 1
fi

# Kill stale processes on the master port
kill $(lsof -ti:29504) 2>/dev/null || true
sleep 1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "  LLM-GC GRPO + PRM Training"
echo "=========================================="
echo "  RL checkpoint  : ${RL_CKPT}"
echo "  PRM checkpoint : ${PRM_CKPT}"
echo "  Output         : ${OUTPUT_DIR}"
echo "  GPUs           : ${GPUS}"
echo "=========================================="

deepspeed --include "localhost:${GPUS}" \
    --master_port 29504 \
    training/rl_train.py \
    --checkpoint_dir "${RL_CKPT}" \
    --prm_checkpoint "${PRM_CKPT}" \
    --w_trace 0.3 \
    --prm_aggregation min \
    --prm_batch_size 16 \
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
    --wandb_project llm-gc-rl-prm \
    --wandb_run_name "grpo-prm-v1"

#!/bin/bash
# Compound-reward GRPO (Phase 3) for the Qwen2.5 backbone variant.
#
# Same PRM checkpoint as the LLaMA pipeline — the PRM is a separate
# Qwen2.5-7B verifier and is independent of the policy backbone.
#
# Usage:
#   bash qwen/script/run_rl_prm.sh [RL_CKPT] [PRM_CKPT] [OUTPUT_DIR] [GPUS]
#
# Examples:
#   bash qwen/script/run_rl_prm.sh
#   bash qwen/script/run_rl_prm.sh \
#       checkpoints/Qwen2.5-3B-Instruct/rl \
#       checkpoints/prm/best \
#       checkpoints/Qwen2.5-3B-Instruct/rl_prm "0,1,2"

set -e

RL_CKPT=${1:-"checkpoints/Qwen2.5-3B-Instruct/rl"}
PRM_CKPT=${2:-"checkpoints/prm/best"}
OUTPUT_DIR=${3:-"checkpoints/Qwen2.5-3B-Instruct/rl_prm"}
GPUS=${4:-"0,1,2"}

if [ ! -f "${PRM_CKPT}/prm_config.json" ]; then
    echo "ERROR: PRM checkpoint not found at ${PRM_CKPT}"
    echo "       (expected ${PRM_CKPT}/prm_config.json)"
    exit 1
fi

kill $(lsof -ti:29514) 2>/dev/null || true
sleep 1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "  LLM-GC GRPO + PRM Training (Qwen backbone)"
echo "=========================================="
echo "  RL checkpoint  : ${RL_CKPT}"
echo "  PRM checkpoint : ${PRM_CKPT}"
echo "  Output         : ${OUTPUT_DIR}"
echo "  GPUs           : ${GPUS}"
echo "=========================================="

deepspeed --include "localhost:${GPUS}" \
    --master_port 29514 \
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
    --wandb_run_name "grpo-prm-qwen-v1"

#!/bin/bash
# SFT for the Qwen2.5 backbone variant.
# Identical pipeline to script/run_sft.sh but:
#   * --model_name  Qwen/Qwen2.5-3B-Instruct
#   * trainer       qwen/training/sft_train.py  (uses qwen.tra for Qwen2 RoPE)
#   * output        checkpoints/Qwen2.5-3B-Instruct/sft
#
# Qwen2.5-3B is smaller than LLaMA-3.1-8B (3B vs 8B params), so with the
# same ZeRO-3 config it needs less GPU memory — we keep the same batch /
# accum defaults for a fair comparison.
#
# Usage:
#   bash qwen/script/run_sft.sh [OUTPUT_DIR] [GPUS] [MODEL_NAME] [RUN_NAME]
#
# Examples:
#   bash qwen/script/run_sft.sh
#   bash qwen/script/run_sft.sh checkpoints/Qwen2.5-3B-Instruct/sft "0,1,2"
#   bash qwen/script/run_sft.sh checkpoints/Qwen2.5-7B-Instruct/sft "0,1,2" \
#                               Qwen/Qwen2.5-7B-Instruct sft-qwen2.5-7b-tra

set -e

OUTPUT_DIR=${1:-"checkpoints/Qwen2.5-3B-Instruct/sft"}
GPUS=${2:-"0,1,2"}
MODEL_NAME=${3:-"Qwen/Qwen2.5-3B-Instruct"}
RUN_NAME=${4:-"sft-qwen2.5-3b-tra"}

# Kill stale processes on the master port
kill $(lsof -ti:29511) 2>/dev/null || true
sleep 1

deepspeed --include "localhost:${GPUS}" \
    --master_port 29511 \
    qwen/training/sft_train.py \
    --model_name "${MODEL_NAME}" \
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
    --output_dir "${OUTPUT_DIR}" \
    --wandb_project llm-gc-sft \
    --wandb_run_name "${RUN_NAME}"

#!/bin/bash
# Evaluation wrapper for Qwen-backbone checkpoints.
#
# Delegates to the existing script/run_eval.sh — evaluation is fully
# backbone-agnostic (evaluate.py uses AutoTokenizer + AutoModel and
# does not apply TRA at inference).  This wrapper exists only to set
# a Qwen-aware default checkpoint path.
#
# Usage:
#   bash qwen/script/run_eval.sh [CHECKPOINT_DIR] [TASKS] [BACKEND] [GPU/TP]
#   bash qwen/script/run_eval.sh --vllm [CHECKPOINT_DIR] [TASKS] [TP]
#
# Examples:
#   bash qwen/script/run_eval.sh --vllm checkpoints/Qwen2.5-3B-Instruct/sft all
#   bash qwen/script/run_eval.sh --vllm checkpoints/Qwen2.5-3B-Instruct/rl all
#   bash qwen/script/run_eval.sh --vllm checkpoints/Qwen2.5-3B-Instruct/rl_prm all

set -e

if [ "${1}" = "--vllm" ]; then
    shift
    CHECKPOINT_DIR=${1:-"checkpoints/Qwen2.5-3B-Instruct/sft"}
    exec bash script/run_eval.sh --vllm "${CHECKPOINT_DIR}" "${@:2}"
else
    CHECKPOINT_DIR=${1:-"checkpoints/Qwen2.5-3B-Instruct/sft"}
    exec bash script/run_eval.sh "${CHECKPOINT_DIR}" "${@:2}"
fi

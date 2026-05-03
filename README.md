# TRACER: Large Language Models as Graph Computational Solvers via Topological Residual Attention

TRACER (Topological Residual Attention for Computational Execution and Reasoning on Graphs) trains LLMs to natively solve graph computational problems through architecture modification and multi-phase training. Built on LLaMA-3.1-8B-Instruct with Topology-Aware Residual Attention (TRA) and evaluated on GraphInstruct (9 tasks).

## Hardware

All training and evaluation were performed on a workstation with **3 ×
NVIDIA RTX PRO 6000 Blackwell** GPUs (≈ 95 GB HBM each, SM 12.0). The
DeepSpeed ZeRO configurations, batch sizes, and `--num-gpus` defaults in
the launch scripts assume this 3-GPU layout. The pipeline can be run on a
different GPU count by passing the GPU list to each script (`bash
script/run_sft.sh OUTPUT "0,1"`), but the per-GPU memory budget assumes
≥ 80 GB cards.

## Installation

```bash
bash script/install.sh
```

This installs PyTorch 2.10 (cu128), HuggingFace `transformers` /
`accelerate` / `peft` / `trl`, `deepspeed`, `flash-attn` 2.7.4.post1
(SM120-compatible), `vllm` (cu128 index), `flashinfer`, plus
`llama-factory` for optional baseline runs. The script also performs a
GPU-detection check (3 visible CUDA devices) and a final dependency-graph
check via `pip check`. See the inline comments in
[`script/install.sh`](script/install.sh) for the exact pinned versions.

## Dataset Preprocessing

Builds the augmented training set from the upstream
[`GraphWiz/GraphInstruct`](https://huggingface.co/datasets/GraphWiz/GraphInstruct)
dataset. Three preprocessors are exposed by `dataset/generate_permuted.py`;
each can be re-run independently and is fully deterministic under `--seed`.
Full design notes: [`dataset/README.md`](dataset/README.md).

```bash
# 1. GraphInstruct-Permuted — k permuted copies per sample
#    (node relabeling + edge-list shuffle + trace re-mapping)
python dataset/generate_permuted.py permuted --k 4 --seed 42

# 2. GraphInstruct-Aug — original + permuted combined (108,750 samples)
python dataset/generate_permuted.py aug --seed 42

# 3. GraphInstruct-RFT-Aug — used by SFT and GRPO/PRM in this submission.
#    71,705 RFT samples × (1 anchor + 2 perms) ≈ 215K samples;
#    --no_graph_repr drops the special token (eliminates train/eval mismatch;
#    L_repr uses the last prompt token instead).
python dataset/generate_permuted.py rft --k 2 --seed 42 --no_graph_repr

# 4. PRM training data — (prefix, step) pairs with synthetic corruptions,
#    derived from RFT traces above. Splits into train.jsonl + val.jsonl.
bash script/run_prm_data.sh
```

Outputs:

- `dataset/GraphInstruct-Permuted/train.jsonl` — Step 1
- `dataset/GraphInstruct-Aug/train.jsonl` — Step 2 (legacy SFT input)
- `dataset/GraphInstruct-RFT-Aug/train.jsonl` — Step 3 (current SFT/RL input)
- `dataset/PRM/{train,val}.jsonl` — Step 4 (PRM training input)

## Training Pipeline

### Phase 1: SFT with TRA

```bash
# Fine-tune with compound loss (L_task + L_repr)
# Requires dataset/GraphInstruct-RFT-Aug from preprocessing step 3 above.
bash script/run_sft.sh
```

### Phase 2: GRPO (RL)

```bash
# RL fine-tuning with shaped reward (from SFT checkpoint)
bash script/run_rl.sh checkpoints/sft checkpoints/rl "0,1,2"
```

### Phase 3: PRM-Augmented GRPO (step-level signal)

```bash
# (a) Train a Qwen-2.5-7B PRM with LoRA + classification head + BCE
#     (PRM training data was produced by preprocessing step 4 above.)
bash script/run_prm_train.sh "0,1,2"

# (b) Second GRPO pass with compound reward (shaped + w_trace * PRM_trace_score)
bash script/run_rl_prm.sh checkpoints/rl checkpoints/prm/best checkpoints/rl_prm "0,1,2"
```

### Evaluation

**Recommended (strict GraphWiz protocol):**

```bash
bash script/run_eval.sh --strict checkpoints/sft
bash script/run_eval.sh --strict checkpoints/rl
bash script/run_eval.sh --strict checkpoints/rl_prm
```

This uses `evaluation/evaluate_strict.py` with `max_new_tokens=1024`,
`--truncate_first_answer`, vLLM greedy decoding, and pure GraphWiz `check()`.

**Legacy modes (skip-on-no-### scoring):**

```bash
bash script/run_eval.sh --vllm checkpoints/sft all
bash script/run_eval.sh --vllm checkpoints/rl cycle              # single task
bash script/run_eval.sh checkpoints/rl cycle hf 0                # HF backend
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate.py \
    --checkpoint_dir checkpoints/rl --tasks cycle \
    --backend hf --deterministic                                  # reproducible
```

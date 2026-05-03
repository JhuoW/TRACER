# Dataset Pipeline

This directory contains the scripts and generated data for building the **GraphInstruct-Aug** training set from the upstream [GraphInstruct](https://huggingface.co/datasets/GraphWiz/GraphInstruct) dataset.

## Quick Start

```bash
# From the project root:

# Step 1 — Generate GraphInstruct-Permuted (node relabeling + edge shuffle + trace remap)
python dataset/generate_permuted.py permuted

# Step 2 — Combine original + permuted into the final training set
python dataset/generate_permuted.py aug
```

The final training set is written to `dataset/GraphInstruct-Aug/train.jsonl`.

## Pipeline Overview

```
HuggingFace: GraphWiz/GraphInstruct (18,125 samples)
    │
    ├──► Step 1: permuted ──► GraphInstruct-Permuted/train.jsonl (90,625 samples)
    │        • 1 original copy (with [GRAPH_REPR] inserted)
    │        • 4 permuted copies per sample (node relabeling + edge shuffle)
    │
    └──► Step 2: aug ──► GraphInstruct-Aug/train.jsonl (108,750 samples)
             • 18,125 original samples (with [GRAPH_REPR] inserted, no permutation)
             • 90,625 permuted samples from Step 1
```

## Step 1: Generate GraphInstruct-Permuted

```bash
python dataset/generate_permuted.py permuted [--k 4] [--seed 42] [--output dataset/GraphInstruct-Permuted]
```

| Argument     | Default                            | Description                          |
| ------------ | ---------------------------------- | ------------------------------------ |
| `--k`      | 4                                  | Number of permuted copies per sample |
| `--seed`   | 42                                 | Random seed for reproducibility      |
| `--output` | `dataset/GraphInstruct-Permuted` | Output directory                     |

**What it does (per `model_components/01_data_augmentation.md`):**

1. **Node relabeling** — Applies a random bijection on node IDs {0..n-1}. All references in the query (edges, question nodes) and answer (CoT trace, final answer) are consistently remapped.
2. **Edge-list shuffle** — Randomizes the serialization order of edges.
3. **Trace re-mapping** — Replaces `node X` references, edge tuples `(X,Y)`, and path lists `[X,Y,Z]` in the reasoning trace using a placeholder-based approach to avoid double-replacement.
4. **`[GRAPH_REPR]` insertion** — Inserts a `[GRAPH_REPR]` special token at the boundary between the graph description and the task question in every sample.

**Output format** (JSONL, one JSON object per line):

```json
{
    "query": "Determine whether ... Q: The nodes ... [GRAPH_REPR] Is there a path ...",
    "answer": " Node 5 and node 3 are connected ... ### Yes.",
    "task": "connectivity",
    "split": "train",
    "original_index": 0,
    "augmentation": "perm_0"
}
```

The `augmentation` field is `"original"` for the unmodified copy (with `[GRAPH_REPR]` only) and `"perm_0"` through `"perm_3"` for permuted copies.

## Step 2: Generate GraphInstruct-Aug

```bash
python dataset/generate_permuted.py aug [--seed 42] [--permuted dataset/GraphInstruct-Permuted] [--output dataset/GraphInstruct-Aug]
```

| Argument       | Default                            | Description           |
| -------------- | ---------------------------------- | --------------------- |
| `--seed`     | 42                                 | Random seed           |
| `--permuted` | `dataset/GraphInstruct-Permuted` | Path to Step 1 output |
| `--output`   | `dataset/GraphInstruct-Aug`      | Output directory      |

This combines:

- The **original** GraphInstruct samples with `[GRAPH_REPR]` inserted (`augmentation="none"`)
- All samples from GraphInstruct-Permuted (`augmentation="original"` and `"perm_0"` through `"perm_3"`)

## Output Summary

| Dataset                                | Samples | Description                                                   |
| -------------------------------------- | ------- | ------------------------------------------------------------- |
| `GraphInstruct-Permuted/train.jsonl` | 90,625  | 18,125 originals + 72,500 permuted (with `[GRAPH_REPR]`)    |
| `GraphInstruct-Aug/train.jsonl`      | 108,750 | 18,125 original + 90,625 permuted (all with `[GRAPH_REPR]`) |

|  |  |
| - | - |

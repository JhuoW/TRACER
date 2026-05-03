#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from training.prm.step_parser import parse_trace_into_steps
from training.prm.step_corruptor import corrupt_step


_NODE_RE = re.compile(r"\bnode (\d+)\b", re.IGNORECASE)
_EDGE_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*\d+)?\s*\)")


def extract_graph_nodes(query: str) -> set[int]:
    nodes: set[int] = set()
    for m in _NODE_RE.finditer(query):
        nodes.add(int(m.group(1)))
    for m in _EDGE_RE.finditer(query):
        nodes.add(int(m.group(1)))
        nodes.add(int(m.group(2)))
    return nodes


def generate_prm_examples(
    sample: dict,
    neg_ratio: float = 1.0,
    rng: random.Random | None = None,
) -> list[dict]:
    if rng is None:
        rng = random.Random()

    query = sample["query"]
    answer = sample["answer"]
    task = sample["task"]

    steps = parse_trace_into_steps(answer, task)
    if not steps:
        return []

    graph_nodes = extract_graph_nodes(query)
    max_node_id = max(graph_nodes) if graph_nodes else 50

    examples: list[dict] = []

    n_full = int(neg_ratio)
    frac = neg_ratio - n_full

    for step in steps:
        examples.append({
            "query": query,
            "prefix": step["prefix"],
            "step": step["text"],
            "label": 1,
            "task": task,
            "corruption_type": "none",
        })

        n_neg = n_full + (1 if rng.random() < frac else 0)
        seen_corruptions: set[str] = set()
        attempts = 0
        while len(seen_corruptions) < n_neg and attempts < n_neg * 3:
            attempts += 1
            corrupted, ctype = corrupt_step(
                step["text"], task,
                max_node_id=max_node_id,
                rng=rng,
                graph_nodes=graph_nodes or None,
            )
            if ctype == "none" or corrupted == step["text"]:
                continue
            if corrupted in seen_corruptions:
                continue
            seen_corruptions.add(corrupted)
            examples.append({
                "query": query,
                "prefix": step["prefix"],
                "step": corrupted,
                "label": 0,
                "task": task,
                "corruption_type": ctype,
            })

    return examples


def main():
    p = argparse.ArgumentParser(description="Generate PRM training data")
    p.add_argument("--input", type=str,
                   default="dataset/GraphInstruct-RFT-Aug/train.jsonl")
    p.add_argument("--output", type=str, default=None,
                   help="(Deprecated) single output file path. Use --output_dir.")
    p.add_argument("--output_dir", type=str, default="dataset/PRM")
    p.add_argument("--neg_ratio", type=float, default=1.0,
                   help="Negative examples per positive example (fractional OK)")
    p.add_argument("--val_frac", type=float, default=0.05,
                   help="Fraction of source traces held out for validation")
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)

    print(f"Loading {args.input}...")
    samples = []
    with open(args.input) as f:
        for line in f:
            s = json.loads(line)
            if s["augmentation"] == "none":
                samples.append(s)
    print(f"  Loaded {len(samples):,} anchor samples")

    rng.shuffle(samples)
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
        print(f"  Capped to {len(samples):,} samples")

    val_size = int(len(samples) * args.val_frac)
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]
    print(f"  Split: {len(train_samples):,} train / {len(val_samples):,} val")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.output is not None:
        train_path = args.output
        val_path = os.path.join(
            os.path.dirname(args.output) or ".", "val.jsonl",
        )
    else:
        train_path = os.path.join(args.output_dir, "train.jsonl")
        val_path = os.path.join(args.output_dir, "val.jsonl")

    def _build(source_samples: list[dict], out_path: str, label: str) -> None:
        all_examples: list[dict] = []
        task_counts: dict[str, dict[str, int]] = {}
        skipped = 0

        for i, sample in enumerate(source_samples):
            examples = generate_prm_examples(
                sample, neg_ratio=args.neg_ratio, rng=rng,
            )
            if not examples:
                skipped += 1
                continue
            all_examples.extend(examples)

            task = sample["task"]
            tc = task_counts.setdefault(
                task, {"positive": 0, "negative": 0, "samples": 0},
            )
            tc["samples"] += 1
            for ex in examples:
                tc["positive" if ex["label"] == 1 else "negative"] += 1

            if (i + 1) % 5000 == 0:
                print(f"  [{label}] {i+1:,}/{len(source_samples):,} samples, "
                      f"{len(all_examples):,} examples")

        rng.shuffle(all_examples)
        with open(out_path, "w") as f:
            for ex in all_examples:
                f.write(json.dumps(ex) + "\n")

        n_pos = sum(1 for ex in all_examples if ex["label"] == 1)
        n_neg = len(all_examples) - n_pos
        print(f"\n[{label}] {len(all_examples):,} examples "
              f"({n_pos:,} pos / {n_neg:,} neg), "
              f"{skipped:,} traces skipped")
        for task in sorted(task_counts):
            tc = task_counts[task]
            print(f"  {task:15s}: {tc['samples']:>6,} traces -> "
                  f"{tc['positive']:>7,} pos + {tc['negative']:>7,} neg")
        print(f"[{label}] saved to {out_path}")

    _build(train_samples, train_path, "train")
    if val_samples:
        _build(val_samples, val_path, "val")


if __name__ == "__main__":
    main()

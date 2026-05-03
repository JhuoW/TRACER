#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.evaluate import (
    ALPACA_PROMPT, check, get_answer, get_query, load_test_data,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="checkpoints/sft")
    parser.add_argument("--task", default="cycle")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--penalties", type=float, nargs="+",
                        default=[1.0, 1.05, 1.1, 1.15])
    parser.add_argument("--output_dir", default="sft_results_strict/sft/rep_penalty_sweep")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    data = load_test_data(args.task)
    queries = [get_query(s) for s in data]
    input_strs = [ALPACA_PROMPT.format(query=q) for q in queries]
    print(f"Loaded {len(data)} samples for task {args.task}")

    from vllm import LLM, SamplingParams
    print(f"Loading {args.checkpoint_dir} with vLLM...")
    llm = LLM(
        model=args.checkpoint_dir,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    summary_rows = []
    for rp in args.penalties:
        print(f"\n=== repetition_penalty={rp} ===")
        sp = SamplingParams(
            temperature=0,
            max_tokens=args.max_new_tokens,
            repetition_penalty=rp,
        )
        outputs = llm.generate(input_strs, sp)
        out_strs = [o.outputs[0].text for o in outputs]

        no_sep = 0
        correct = 0
        for sample, out in zip(data, out_strs):
            truth = get_answer(sample)
            pred = out.lstrip()
            if "###" not in pred:
                no_sep += 1
            if check(args.task, truth.lower(), pred.lower()):
                correct += 1
        total = len(data)
        acc = correct / total
        print(f"  Strict acc: {correct}/{total} = {acc*100:.2f}%")
        print(f"  No-### count: {no_sep}/{total}")
        summary_rows.append({
            "repetition_penalty": rp,
            "correct": correct,
            "total": total,
            "accuracy": acc,
            "no_sep": no_sep,
        })

        out_path = Path(args.output_dir) / f"_gen_{args.task}_rp{rp}.jsonl"
        with open(out_path, "w") as f:
            for sample, inp, out in zip(data, input_strs, out_strs):
                json.dump({
                    "source_data": sample,
                    "input_str": inp,
                    "output_str": out,
                    "task": args.task,
                }, f, default=str)
                f.write("\n")

    print("\n" + "=" * 60)
    print(f"REPETITION PENALTY SWEEP — {args.task}")
    print("=" * 60)
    print(f"  {'rp':<6} {'correct':<10} {'acc':<10} {'no-###':<8}")
    for r in summary_rows:
        print(f"  {r['repetition_penalty']:<6} {r['correct']}/{r['total']}    {r['accuracy']*100:.2f}%     {r['no_sep']}")

    with open(Path(args.output_dir) / "summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)


if __name__ == "__main__":
    main()

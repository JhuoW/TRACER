#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.evaluate import (
    ALPACA_PROMPT,
    HF_TASK_NAME,
    TASKS,
    _truncate_at_first_answer,
    check,
    get_answer,
    get_query,
    load_model_hf,
    load_model_vllm,
    load_test_data,
)


def evaluate_task_strict(
    task: str,
    generate_fn,
    args,
) -> tuple[int, int]:
    print(f"\n{'='*60}")
    print(f"Evaluating (strict): {task}")
    print(f"{'='*60}")

    data = load_test_data(task, args.data_dir)
    print(f"  Loaded {len(data)} test samples")

    gen_jsonl = Path(args.output_dir) / f"_gen_{task}_datas.jsonl"
    start_index = 0
    if gen_jsonl.exists():
        with open(gen_jsonl) as f:
            start_index = sum(1 for _ in f)
        print(f"  Resuming from index {start_index}")

    if start_index < len(data):
        remaining = data[start_index:]
        queries = [get_query(s) for s in remaining]
        input_strs = [ALPACA_PROMPT.format(query=q) for q in queries]

        if args.backend == "vllm":
            output_strs = generate_fn(input_strs, args.max_new_tokens)
            for j, (sample, inp, out) in enumerate(
                zip(remaining, input_strs, output_strs)
            ):
                with open(gen_jsonl, "a") as f:
                    json.dump({
                        "index": start_index + j,
                        "source_data": sample,
                        "input_str": inp,
                        "output_str": out,
                        "task": task,
                    }, f, default=str)
                    f.write("\n")
        else:
            from tqdm import tqdm
            for i in tqdm(range(0, len(remaining), args.batch_size), desc=task):
                batch_samples = remaining[i: i + args.batch_size]
                batch_inputs = input_strs[i: i + args.batch_size]
                batch_outputs = generate_fn(batch_inputs, args.max_new_tokens)
                for j, (sample, inp, out) in enumerate(
                    zip(batch_samples, batch_inputs, batch_outputs)
                ):
                    with open(gen_jsonl, "a") as f:
                        json.dump({
                            "index": start_index + i + j,
                            "source_data": sample,
                            "input_str": inp,
                            "output_str": out,
                            "task": task,
                        }, f, default=str)
                        f.write("\n")

    return score_jsonl_strict(
        gen_jsonl, task, args.output_dir,
        truncate_first_answer=args.truncate_first_answer,
    )


def score_jsonl_strict(
    jsonl_path: Path,
    task: str,
    output_dir: str,
    truncate_first_answer: bool = False,
) -> tuple[int, int]:
    with open(jsonl_path) as f:
        gen_datas = [json.loads(line) for line in f]

    correct_results = []
    wrong_results = []
    for gen in gen_datas:
        truth = get_answer(gen["source_data"])
        predict = gen["output_str"].lstrip()
        if truncate_first_answer:
            predict = _truncate_at_first_answer(predict)

        result = {
            **gen,
            "extract_true": truth,
            "extract_pred": predict,
            "is_correct": check(task, truth.lower(), predict.lower()),
        }
        (correct_results if result["is_correct"] else wrong_results).append(result)

    correct = len(correct_results)
    total = len(correct_results) + len(wrong_results)
    accuracy = correct / total if total > 0 else 0.0
    suffix = " [trunc-first]" if truncate_first_answer else ""
    print(f"  Strict accuracy{suffix} = {correct}/{total} = {accuracy:.4f} ({accuracy*100:.2f}%)")

    out_suffix = "strict_trunc" if truncate_first_answer else "strict"
    with open(Path(output_dir) / f"{task}_correct_{out_suffix}.json", "w") as f:
        json.dump(correct_results, f, ensure_ascii=False, indent=2, default=str)
    with open(Path(output_dir) / f"{task}_wrong_{out_suffix}.json", "w") as f:
        json.dump(wrong_results, f, ensure_ascii=False, indent=2, default=str)

    return correct, total


def score_only(
    score_dir: str,
    tasks: list[str],
    truncate_first_answer: bool = False,
) -> tuple[dict[str, float], dict]:
    print(f"Re-scoring existing outputs in: {score_dir}")
    if truncate_first_answer:
        print("  --truncate_first_answer ON")
    results = {}
    counts = {}
    for task in tasks:
        gen_path = Path(score_dir) / f"_gen_{task}_datas.jsonl"
        if not gen_path.exists():
            print(f"  Skipping {task}: {gen_path} not found")
            continue
        correct, total = score_jsonl_strict(
            gen_path, task, score_dir,
            truncate_first_answer=truncate_first_answer,
        )
        results[task] = correct / total if total > 0 else 0.0
        counts[task] = (correct, total)
    return results, counts


def write_report(output_dir: str, results: dict, counts: dict, suffix: str = "strict"):
    average = sum(results.values()) / len(results) if results else 0.0
    print(f"\n{'='*60}")
    print(f"STRICT-GRAPHWIZ RESULTS")
    print(f"{'='*60}")
    for task, acc in results.items():
        c, t = counts[task]
        print(f"  {task:<15} {c}/{t} = {acc*100:.2f}%")
    print(f"  {'Average':<15} {average*100:.2f}%")
    print(f"{'='*60}")

    csv_path = Path(output_dir) / f"eval_results_{suffix}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Task", "Correct", "Total", "Accuracy"])
        for task, acc in results.items():
            c, t = counts[task]
            writer.writerow([task, c, t, f"{acc:.4f}"])
        writer.writerow(["Average", "", "", f"{average:.4f}"])
    print(f"\nResults saved to {csv_path}")

    json_path = Path(output_dir) / f"eval_results_{suffix}.json"
    with open(json_path, "w") as f:
        json.dump({
            "per_task": results,
            "counts": {k: {"correct": v[0], "total": v[1]} for k, v in counts.items()},
            "average": average,
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Strict GraphWiz-protocol evaluator")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Path to checkpoint (LoRA for hf, merged for vllm). "
                             "Required unless --score_only is set.")
    parser.add_argument("--score_only", type=str, default=None,
                        help="Path to an existing run dir with _gen_*.jsonl files. "
                             "Re-scores using strict protocol; no GPU/inference.")
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--tasks", type=str, nargs="+", default=None)
    parser.add_argument("--backend", type=str, default="hf", choices=["hf", "vllm"])
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--truncate_first_answer", action="store_true",
                        help="Cut prediction at first '###' answer before scoring. "
                             "Avoids the 'second answer' regression on numeric tasks "
                             "when using long --max_new_tokens. Slight deviation from "
                             "pure GraphWiz protocol but defensible.")
    args = parser.parse_args()

    tasks = args.tasks if args.tasks else TASKS

    if args.score_only:
        if not os.path.isdir(args.score_only):
            raise SystemExit(f"--score_only path not found: {args.score_only}")
        args.output_dir = args.score_only
        results, counts = score_only(
            args.score_only, tasks,
            truncate_first_answer=args.truncate_first_answer,
        )
        suffix = "strict_trunc" if args.truncate_first_answer else "strict"
        write_report(args.output_dir, results, counts, suffix=suffix)
        return

    if not args.checkpoint_dir:
        raise SystemExit("Provide either --checkpoint_dir or --score_only.")

    if args.output_dir is None:
        base_model_name = args.base_model
        if base_model_name is None:
            adapter_cfg_path = os.path.join(args.checkpoint_dir, "adapter_config.json")
            if os.path.exists(adapter_cfg_path):
                with open(adapter_cfg_path) as f:
                    base_model_name = json.load(f)["base_model_name_or_path"]
            else:
                name_file = os.path.join(args.checkpoint_dir, "base_model_name.txt")
                if os.path.exists(name_file):
                    with open(name_file) as f:
                        base_model_name = f.read().strip()
                else:
                    base_model_name = args.checkpoint_dir
        model_short = base_model_name.rstrip("/").split("/")[-1]
        task_dir = "_".join(tasks) if len(tasks) <= 3 else "all"
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = os.path.join("sft_results_strict", model_short, task_dir, timestamp)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {args.output_dir}")
    print(f"Backend: {args.backend}")
    with open(Path(args.output_dir) / "eval_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    if args.backend == "vllm":
        generate_fn = load_model_vllm(args)
    else:
        generate_fn = load_model_hf(args)

    results = {}
    counts = {}
    for task in tasks:
        c, t = evaluate_task_strict(task, generate_fn, args)
        results[task] = c / t if t > 0 else 0.0
        counts[task] = (c, t)

    suffix = "strict_trunc" if args.truncate_first_answer else "strict"
    write_report(args.output_dir, results, counts, suffix=suffix)


if __name__ == "__main__":
    main()

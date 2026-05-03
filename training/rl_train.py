#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import deepspeed
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training.rewards import compute_reward, compute_shaped_reward


def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg, flush=True)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, tokenizer, args):
    model_to_save = model.module if hasattr(model, "module") else model
    if args.global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        with open(os.path.join(args.output_dir, "base_model_name.txt"), "w") as f:
            f.write(args.checkpoint_dir)
    print_rank_0(f"Checkpoint saved to {args.output_dir}", args.global_rank)


ALPACA_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request step by step.\n\n"
    "### Instruction:\n{query}\n\n### Response:"
)


class GRPODataset(Dataset):

    def __init__(self, data_path: str, tasks: list[str] | None = None,
                 max_samples: int | None = None, seed: int = 42):
        self.samples = []
        with open(data_path) as f:
            for line in f:
                row = json.loads(line)
                if row["augmentation"] != "none":
                    continue
                if tasks and row["task"] not in tasks:
                    continue
                self.samples.append(row)

        if max_samples and max_samples < len(self.samples):
            rng = random.Random(seed)
            rng.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {
            "prompt": ALPACA_PROMPT.format(query=s["query"]),
            "query":  s["query"],
            "answer": s["answer"],
            "task":   s["task"],
        }


class GRPOCollator:

    def __call__(self, batch: list[dict]) -> dict:
        return {
            "prompts": [b["prompt"] for b in batch],
            "queries": [b["query"]  for b in batch],
            "answers": [b["answer"] for b in batch],
            "tasks":   [b["task"]   for b in batch],
        }


def _logprobs_chunk(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                    use_cache=False)
    logits = outputs.logits[:, :-1, :]
    del outputs
    labels = input_ids[:, 1:]

    per_token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    ).reshape(labels.shape)

    return -per_token_loss


def compute_token_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    chunk_size: int = 2,
) -> torch.Tensor:
    B = input_ids.shape[0]
    if B <= chunk_size:
        return _logprobs_chunk(model, input_ids, attention_mask)

    chunks = []
    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        chunk_lp = _logprobs_chunk(
            model, input_ids[start:end], attention_mask[start:end],
        )
        chunks.append(chunk_lp)
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def generate_completions(
    model,
    tokenizer,
    prompts: list[str],
    num_generations: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    B = len(prompts)
    G = num_generations

    tokenizer.padding_side = "left"
    prompt_enc = tokenizer(
        prompts, padding=True, return_tensors="pt",
        truncation=True, max_length=1024,
    )
    prompt_ids = prompt_enc.input_ids
    prompt_mask = prompt_enc.attention_mask
    prompt_len = prompt_ids.shape[1]

    repeated_ids = prompt_ids.repeat_interleave(G, dim=0).to(device)
    repeated_mask = prompt_mask.repeat_interleave(G, dim=0).to(device)

    gen_model = model.module if hasattr(model, "module") else model
    generated_ids = gen_model.generate(
        input_ids=repeated_ids,
        attention_mask=repeated_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
    )

    del repeated_ids, repeated_mask

    full_ids = generated_ids
    BG, T = full_ids.shape

    full_mask = torch.ones(BG, T, dtype=torch.long, device=device)
    full_mask[:, :prompt_len] = prompt_mask.repeat_interleave(G, dim=0).to(device)

    completion_mask = torch.zeros(BG, T, dtype=torch.float, device=device)
    for i in range(BG):
        comp_tokens = full_ids[i, prompt_len:]
        eos_positions = (comp_tokens == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            end_idx = eos_positions[0].item() + 1
        else:
            end_idx = comp_tokens.shape[0]
        completion_mask[i, prompt_len:prompt_len + end_idx] = 1.0
        full_mask[i, prompt_len + end_idx:] = 0

    return full_ids, full_mask, completion_mask, prompt_len


def compute_grpo_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    epsilon: float = 0.2,
) -> torch.Tensor:
    ratio = (new_logprobs - old_logprobs).exp()

    adv_per_token = advantages.unsqueeze(-1) * completion_mask

    surr1 = ratio * adv_per_token
    surr2 = ratio.clamp(1.0 - epsilon, 1.0 + epsilon) * adv_per_token
    loss = -torch.min(surr1, surr2).sum() / completion_mask.sum().clamp(min=1)
    return loss


def get_ds_config(args, world_size):
    micro_bs = args.batch_size * args.num_generations
    return {
        "train_micro_batch_size_per_gpu": micro_bs,
        "train_batch_size": micro_bs * world_size * args.gradient_accumulation_steps,
        "steps_per_print": args.logging_steps * 10,
        "zero_optimization": {
            "stage": 2,
            "offload_param": {"device": "none"},
            "offload_optimizer": {"device": "none"},
        },
        "bf16": {"enabled": True},
        "gradient_clipping": args.max_grad_norm,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }


def parse_args():
    p = argparse.ArgumentParser(description="GRPO training (DeepSpeed)")

    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/sft",
                   help="Path to the SFT checkpoint to start from")
    p.add_argument("--attn_implementation", type=str, default="flash_attention_2")

    p.add_argument("--data_path", type=str,
                   default="dataset/GraphInstruct-RFT-Aug/train.jsonl")
    p.add_argument("--tasks", type=str, nargs="*", default=None,
                   help="Filter to specific tasks (e.g., cycle connectivity)")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap number of training prompts (default: use all)")

    p.add_argument("--num_generations", type=int, default=8,
                   help="G: completions per prompt")
    p.add_argument("--num_iterations", type=int, default=1,
                   help="mu: policy update iterations per generation batch")
    p.add_argument("--epsilon", type=float, default=0.2,
                   help="PPO clip range")
    p.add_argument("--reward_type", type=str, default="shaped",
                   choices=["binary", "shaped"],
                   help="Reward function: 'binary' (0/1) or 'shaped' "
                        "(format bonus + numeric partial credit)")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Sampling temperature for generation (1.0 recommended "
                        "for diversity — lower values cause all G completions "
                        "to converge, killing GRPO signal)")
    p.add_argument("--top_p", type=float, default=0.95,
                   help="Top-p sampling")
    p.add_argument("--max_new_tokens", type=int, default=1024,
                   help="Max completion tokens during generation")

    p.add_argument("--prm_checkpoint", type=str, default=None,
                   help="Path to trained PRM checkpoint (e.g. checkpoints/prm/best). "
                        "If set, reward = (1-w_trace)*R_shaped + w_trace*R_PRM.")
    p.add_argument("--w_trace", type=float, default=0.3,
                   help="Weight of PRM trace score in compound reward")
    p.add_argument("--prm_aggregation", type=str, default="min",
                   choices=["min", "mean", "product"],
                   help="How to aggregate PRM step scores into a trace score")
    p.add_argument("--prm_batch_size", type=int, default=16,
                   help="Inner batch size for PRM step scoring")

    p.add_argument("--batch_size", type=int, default=1,
                   help="Prompts per GPU per step (keep low — each prompt "
                        "generates G completions, all forwarded with gradients)")
    p.add_argument("--gradient_accumulation_steps", type=int, default=4,
                   help="Accumulate gradients over this many generation batches")
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=5e-7)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--local_rank", type=int, default=-1)

    p.add_argument("--output_dir", type=str, default="checkpoints/rl")
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--save_steps", type=int, default=500,
                   help="Save checkpoint every N optimizer steps (0 = end only)")

    p.add_argument("--wandb_project", type=str, default="llm-gc-rl")
    p.add_argument("--wandb_run_name", type=str, default="grpo-v1")

    p = deepspeed.add_config_arguments(p)
    return p.parse_args()


def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()

    args.global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    set_random_seed(args.seed)

    if args.global_rank == 0:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                       config=vars(args))
        except ImportError:
            wandb = None
            print_rank_0("wandb not installed, skipping")
    else:
        wandb = None

    print_rank_0(f"Loading model from {args.checkpoint_dir}...", args.global_rank)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_rank_0(f"Model loaded: {trainable:,} trainable params", args.global_rank)

    prm_scorer = None
    if args.prm_checkpoint:
        from training.prm.prm_scorer import PRMScorer
        print_rank_0(
            f"Loading PRM from {args.prm_checkpoint} "
            f"(w_trace={args.w_trace}, agg={args.prm_aggregation}, "
            f"bs={args.prm_batch_size})...",
            args.global_rank,
        )
        prm_scorer = PRMScorer(
            args.prm_checkpoint,
            device=device,
            batch_size=args.prm_batch_size,
        )
        print_rank_0("  PRM ready.", args.global_rank)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    print_rank_0("Loading dataset...", args.global_rank)
    dataset = GRPODataset(
        args.data_path,
        tasks=args.tasks,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    print_rank_0(f"Dataset: {len(dataset):,} prompts", args.global_rank)

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=args.global_rank,
        shuffle=True, seed=args.seed,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        collate_fn=GRPOCollator(), num_workers=2, pin_memory=True,
        drop_last=True,
    )

    optimizer = FusedAdam(
        model.parameters(), lr=args.learning_rate,
        betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )

    steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    total_opt_steps = steps_per_epoch * args.num_train_epochs * args.num_iterations
    lr_scheduler = get_scheduler(
        name="cosine", optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_opt_steps,
    )

    ds_config = get_ds_config(args, world_size)
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model, optimizer=optimizer, config=ds_config,
        lr_scheduler=lr_scheduler, args=args, dist_init_required=True,
    )

    prm_scorer = None
    if args.prm_checkpoint:
        from training.prm.prm_scorer import PRMScorer
        print_rank_0(
            f"Loading PRM from {args.prm_checkpoint} "
            f"(w_trace={args.w_trace}, agg={args.prm_aggregation}, "
            f"bs={args.prm_batch_size})...",
            args.global_rank,
        )
        prm_scorer = PRMScorer(
            args.prm_checkpoint,
            device=device,
            batch_size=args.prm_batch_size,
        )
        if dist.is_initialized():
            dist.barrier()
        print_rank_0("  PRM ready on all ranks.", args.global_rank)

    print_rank_0("***** Running GRPO training *****", args.global_rank)
    print_rank_0(f"  Checkpoint       = {args.checkpoint_dir}", args.global_rank)
    print_rank_0(f"  Prompts          = {len(dataset):,}", args.global_rank)
    print_rank_0(f"  Batch size       = {args.batch_size} prompts/GPU", args.global_rank)
    print_rank_0(f"  Generations (G)  = {args.num_generations}", args.global_rank)
    print_rank_0(f"  Iterations (mu)  = {args.num_iterations}", args.global_rank)
    print_rank_0(f"  Grad accum steps = {args.gradient_accumulation_steps}", args.global_rank)
    print_rank_0(f"  Total opt steps  = {total_opt_steps}", args.global_rank)
    print_rank_0(f"  Temperature      = {args.temperature}", args.global_rank)
    print_rank_0(f"  Epsilon          = {args.epsilon}", args.global_rank)
    print_rank_0(f"  Reward type      = {args.reward_type}", args.global_rank)
    if prm_scorer is not None:
        print_rank_0(
            f"  PRM              = {args.prm_checkpoint}  "
            f"(w_trace={args.w_trace}, agg={args.prm_aggregation})",
            args.global_rank,
        )
    print_rank_0(f"  Learning rate    = {args.learning_rate}", args.global_rank)

    reward_fn = compute_shaped_reward if args.reward_type == "shaped" else compute_reward

    G = args.num_generations
    global_step = 0
    opt_step = 0

    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"\n{'='*60}\n"
            f"Epoch {epoch + 1}/{args.num_train_epochs}\n"
            f"{'='*60}",
            args.global_rank,
        )
        sampler.set_epoch(epoch)

        epoch_rewards = []
        epoch_correct = []
        epoch_loss = []
        epoch_signal = []
        task_rewards: dict[str, list[float]] = {}

        epoch_prm_scores: list[float] = []
        epoch_shaped: list[float] = []

        for step, batch in enumerate(dataloader):
            prompts = batch["prompts"]
            queries = batch["queries"]
            answers = batch["answers"]
            tasks = batch["tasks"]
            B = len(prompts)

            step_t0 = time.time()

            model.eval()
            full_ids, full_mask, comp_mask, prompt_len = generate_completions(
                model, tokenizer, prompts,
                num_generations=G,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
            )

            with torch.no_grad():
                old_logprobs = compute_token_logprobs(
                    model, full_ids, full_mask,
                ).cpu()

            comp_mask_shifted = comp_mask[:, 1:]

            torch.cuda.empty_cache()

            completion_texts: list[str] = []
            for i in range(B * G):
                comp_ids = full_ids[i, prompt_len:]
                mask_i = comp_mask[i, prompt_len:]
                completion_texts.append(tokenizer.decode(
                    comp_ids[mask_i.bool()], skip_special_tokens=True,
                ))

            shaped_rewards = [0.0] * (B * G)
            correctness_arr = [0.0] * (B * G)
            for i, comp_text in enumerate(completion_texts):
                prompt_idx = i // G
                shaped_rewards[i] = reward_fn(
                    tasks[prompt_idx], answers[prompt_idx], comp_text,
                )
                correctness_arr[i] = float(
                    compute_reward(tasks[prompt_idx], answers[prompt_idx], comp_text)
                )

            if prm_scorer is not None:
                per_comp_queries = [queries[i // G] for i in range(B * G)]
                per_comp_tasks   = [tasks[i // G]   for i in range(B * G)]
                prm_trace_scores = prm_scorer.score_traces_batch(
                    per_comp_queries, completion_texts, per_comp_tasks,
                    aggregation=args.prm_aggregation,
                )
                w = args.w_trace
                combined = [
                    (1.0 - w) * r_s + w * r_t
                    for r_s, r_t in zip(shaped_rewards, prm_trace_scores)
                ]
            else:
                prm_trace_scores = None
                combined = shaped_rewards

            rewards = torch.tensor(combined, dtype=torch.float32, device=device)
            correctness = torch.tensor(correctness_arr, dtype=torch.float32)

            rewards_grouped = rewards.view(B, G)
            mean_r = rewards_grouped.mean(dim=1, keepdim=True)
            std_r = rewards_grouped.std(dim=1, keepdim=True)
            advantages = ((rewards_grouped - mean_r) / (std_r + 1e-4)).view(-1)

            has_signal = (std_r.squeeze(-1) > 1e-6).any().item()

            mean_reward = rewards.mean().item()
            correct_frac = correctness.mean().item()
            shaped_mean = float(np.mean(shaped_rewards)) if shaped_rewards else 0.0
            prm_mean = (
                float(np.mean(prm_trace_scores))
                if prm_trace_scores is not None else None
            )
            epoch_rewards.append(mean_reward)
            epoch_correct.append(correct_frac)
            epoch_signal.append(float(has_signal))
            epoch_shaped.append(shaped_mean)
            if prm_mean is not None:
                epoch_prm_scores.append(prm_mean)
            for pi in range(B):
                t = tasks[pi]
                task_rewards.setdefault(t, []).append(correctness[pi * G:(pi + 1) * G].mean().item())

            model.train()
            for mu in range(args.num_iterations):
                new_logprobs = compute_token_logprobs(
                    model, full_ids, full_mask,
                )

                loss = compute_grpo_loss(
                    new_logprobs, old_logprobs.to(device),
                    advantages, comp_mask_shifted,
                    epsilon=args.epsilon,
                )

                model.backward(loss)
                model.step()

                epoch_loss.append(loss.item())

            global_step += 1

            if global_step % args.gradient_accumulation_steps == 0:
                opt_step += 1

            if step % args.logging_steps == 0:
                step_time = time.time() - step_t0
                window = args.logging_steps
                avg_reward = np.mean(epoch_rewards[-window:]) if epoch_rewards else 0
                avg_correct = np.mean(epoch_correct[-window:]) if epoch_correct else 0
                avg_loss = np.mean(epoch_loss[-window:]) if epoch_loss else 0
                signal_rate = np.mean(epoch_signal[-window:]) if epoch_signal else 0
                lr = lr_scheduler.get_last_lr()[0]

                prm_str = ""
                if epoch_prm_scores:
                    prm_str = f"  prm={np.mean(epoch_prm_scores[-window:]):.3f}"
                print_rank_0(
                    f"  step {step:>5}/{len(dataloader)}  "
                    f"reward={avg_reward:.3f}  correct={avg_correct:.1%}{prm_str}  "
                    f"loss={avg_loss:.4f}  lr={lr:.2e}  "
                    f"sig={signal_rate:.0%}  {step_time:.1f}s",
                    args.global_rank,
                )

                if wandb and args.global_rank == 0:
                    log_dict = {
                        "train/reward": mean_reward,
                        "train/reward_shaped": shaped_mean,
                        "train/correct_frac": correct_frac,
                        "train/loss": epoch_loss[-1] if epoch_loss else 0,
                        "train/lr": lr,
                        "train/signal_rate": signal_rate,
                        "train/epoch": epoch + step / len(dataloader),
                    }
                    if prm_mean is not None:
                        log_dict["train/reward_prm"] = prm_mean
                        log_dict["train/w_trace"] = args.w_trace
                    for t, accs in task_rewards.items():
                        if accs:
                            log_dict[f"task/{t}"] = np.mean(accs[-50:])
                    wandb.log(log_dict, step=global_step)

            if (args.save_steps > 0 and opt_step > 0
                    and opt_step % args.save_steps == 0
                    and global_step % args.gradient_accumulation_steps == 0):
                step_dir = os.path.join(args.output_dir, f"step_{opt_step}")
                save_checkpoint(model, tokenizer, argparse.Namespace(
                    output_dir=step_dir, global_rank=args.global_rank,
                    checkpoint_dir=args.checkpoint_dir,
                ))

            del full_ids, full_mask, comp_mask, comp_mask_shifted
            del old_logprobs, rewards, advantages
            torch.cuda.empty_cache()

        avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0
        avg_correct = np.mean(epoch_correct) if epoch_correct else 0
        avg_signal = np.mean(epoch_signal) if epoch_signal else 0
        print_rank_0(
            f"\nEpoch {epoch + 1} done — "
            f"avg reward: {avg_reward:.3f}, "
            f"avg correct: {avg_correct:.1%}, "
            f"signal rate: {avg_signal:.0%}",
            args.global_rank,
        )
        if task_rewards and args.global_rank <= 0:
            print_rank_0("  Per-task accuracy:", args.global_rank)
            for t in sorted(task_rewards):
                acc = np.mean(task_rewards[t])
                n = len(task_rewards[t])
                print_rank_0(f"    {t:<15s} {acc:.1%}  ({n} prompts)", args.global_rank)

    save_checkpoint(model, tokenizer, args)

    if wandb and args.global_rank == 0:
        wandb.finish()

    print_rank_0("GRPO training complete.", args.global_rank)


if __name__ == "__main__":
    main()

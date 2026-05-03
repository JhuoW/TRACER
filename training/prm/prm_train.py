#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import deepspeed
from deepspeed.ops.adam import FusedAdam
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModel, AutoTokenizer, get_scheduler
from peft import LoraConfig, get_peft_model, TaskType


PRM_TEMPLATE = (
    "You are a reasoning step verifier. Given a graph problem, the reasoning "
    "so far, and the current step, determine if the current step is correct.\n\n"
    "### Problem:\n{query}\n\n"
    "### Reasoning so far:\n{prefix}\n\n"
    "### Current step:\n{step}\n\n"
    "### Verdict:"
)


class PRMDataset(Dataset):

    def __init__(self, data_path: str, max_samples: int = -1, seed: int = 42):
        self.samples = []
        with open(data_path) as f:
            for line in f:
                self.samples.append(json.loads(line))
        if max_samples > 0 and len(self.samples) > max_samples:
            rng = random.Random(seed)
            rng.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        text = PRM_TEMPLATE.format(
            query=s["query"],
            prefix=s["prefix"] if s["prefix"] else "(start of reasoning)",
            step=s["step"],
        )
        return {
            "text": text,
            "label": s["label"],
            "task": s["task"],
        }


class PRMCollator:

    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer.truncation_side = "left"

    def __call__(self, batch: list[dict]) -> dict:
        texts = [b["text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }


class PRMHead(nn.Module):

    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1, bias=True)
        nn.init.normal_(self.score.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.score.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.score(hidden_states).squeeze(-1)


def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg, flush=True)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_last_token_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    return attention_mask.sum(dim=1) - 1


def compute_pos_weight(data_path: str) -> float:
    n_pos = n_neg = 0
    with open(data_path) as f:
        for line in f:
            s = json.loads(line)
            if s["label"] == 1:
                n_pos += 1
            else:
                n_neg += 1
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def get_ds_config(args):
    return {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "train_batch_size": (
            args.batch_size
            * max(dist.get_world_size(), 1)
            * args.gradient_accumulation_steps
        ),
        "steps_per_print": args.logging_steps,
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
        },
        "bf16": {"enabled": True},
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Train PRM (Qwen-2.5-7B)")

    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")

    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    p.add_argument("--data_path", type=str, default="dataset/PRM/train.jsonl")
    p.add_argument("--val_data_path", type=str, default="dataset/PRM/val.jsonl",
                   help="Validation set (optional — skipped if missing)")
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--max_samples", type=int, default=-1)

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--num_train_epochs", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--head_lr_mult", type=float, default=10.0,
                   help="Multiplier for PRM head LR vs LoRA LR")
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--use_pos_weight", action="store_true", default=True,
                   help="Weight BCE loss by neg/pos ratio for class balance")

    p.add_argument("--local_rank", type=int, default=-1)

    p.add_argument("--output_dir", type=str, default="checkpoints/prm")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=-1,
                   help="Run validation every N opt steps (-1 = end of epoch only)")

    p.add_argument("--wandb_project", type=str, default="llm-gc-prm")
    p.add_argument("--wandb_run_name", type=str, default="prm-qwen-7b")

    p = deepspeed.add_config_arguments(p)
    return p.parse_args()


def backbone_pool(model, prm_head, input_ids, attention_mask):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    last_hidden = outputs.last_hidden_state

    last_positions = get_last_token_positions(attention_mask)
    pooled = last_hidden[
        torch.arange(last_hidden.size(0), device=last_hidden.device),
        last_positions,
    ]
    logits = prm_head(pooled.to(prm_head.score.weight.dtype))
    return logits


@torch.no_grad()
def run_validation(model, prm_head, val_loader, device, rank: int) -> dict:
    model.eval()
    prm_head.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = backbone_pool(model, prm_head, input_ids, attention_mask)
        loss = F.binary_cross_entropy_with_logits(logits.float(), labels)
        preds = (logits > 0).float()
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

    if dist.is_initialized():
        t = torch.tensor(
            [total_loss, total_correct, total_count],
            device=device, dtype=torch.float64,
        )
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_loss, total_correct, total_count = t.tolist()

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    model.train()
    prm_head.train()
    return {"val_loss": avg_loss, "val_acc": avg_acc, "val_n": int(total_count)}


def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()

    args.global_rank = dist.get_rank() if dist.is_initialized() else 0
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

    ds_config = get_ds_config(args)

    print_rank_0("Loading tokenizer...", args.global_rank)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print_rank_0(f"Loading backbone: {args.model_name}...", args.global_rank)
    model = AutoModel.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print_rank_0(
        f"LoRA: {trainable:,}/{total:,} params trainable "
        f"(r={args.lora_r}, alpha={args.lora_alpha})",
        args.global_rank,
    )

    hidden_size = model.config.hidden_size
    prm_head = PRMHead(hidden_size).to(torch.bfloat16)
    model.add_module("prm_head", prm_head)
    print_rank_0(f"PRM head: {hidden_size} -> 1 (attached as model.prm_head)",
                 args.global_rank)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    print_rank_0("Loading PRM train dataset...", args.global_rank)
    train_dataset = PRMDataset(
        args.data_path, max_samples=args.max_samples, seed=args.seed,
    )
    print_rank_0(f"  train: {len(train_dataset):,} examples", args.global_rank)

    val_dataset = None
    if args.val_data_path and os.path.exists(args.val_data_path):
        val_dataset = PRMDataset(args.val_data_path, max_samples=-1)
        print_rank_0(f"  val:   {len(val_dataset):,} examples", args.global_rank)
    else:
        print_rank_0("  val:   (none — skipping validation)", args.global_rank)

    pos_weight = None
    if args.use_pos_weight:
        pw = compute_pos_weight(args.data_path)
        pos_weight = torch.tensor(pw, device=device, dtype=torch.float32)
        print_rank_0(f"  pos_weight = {pw:.3f}", args.global_rank)

    collator = PRMCollator(tokenizer, max_length=args.max_seq_len)

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=dist.get_world_size(), rank=args.global_rank,
        shuffle=True, seed=args.seed,
    ) if dist.is_initialized() else None
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), collate_fn=collator,
        num_workers=4, pin_memory=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=dist.get_world_size(),
            rank=args.global_rank, shuffle=False,
        ) if dist.is_initialized() else None
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, sampler=val_sampler,
            shuffle=False, collate_fn=collator,
            num_workers=2, pin_memory=True,
        )

    head_param_ids = {id(p) for p in prm_head.parameters()}
    lora_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in head_param_ids
    ]
    head_params = [p for p in prm_head.parameters() if p.requires_grad]
    param_groups = [
        {"params": lora_params, "lr": args.learning_rate},
        {"params": head_params, "lr": args.learning_rate * args.head_lr_mult},
    ]
    optimizer = FusedAdam(param_groups, lr=args.learning_rate, betas=(0.9, 0.95))

    num_update_steps = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    total_steps = args.num_train_epochs * num_update_steps
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model, optimizer=optimizer, args=args,
        config=ds_config, lr_scheduler=lr_scheduler,
        dist_init_required=True,
    )
    prm_head = prm_head.to(device)

    print_rank_0("***** Training PRM *****", args.global_rank)
    print_rank_0(f"  Examples        = {len(train_dataset):,}", args.global_rank)
    print_rank_0(f"  Epochs          = {args.num_train_epochs}", args.global_rank)
    print_rank_0(f"  Batch size      = {args.batch_size}", args.global_rank)
    print_rank_0(f"  Grad accum      = {args.gradient_accumulation_steps}", args.global_rank)
    print_rank_0(f"  Total opt steps = {total_steps:,}", args.global_rank)

    best_val_loss = float("inf")
    global_step = 0
    opt_step = 0

    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"\nEpoch {epoch + 1}/{args.num_train_epochs}", args.global_rank,
        )
        model.train()
        prm_head.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = backbone_pool(model, prm_head, input_ids, attention_mask)

            loss = F.binary_cross_entropy_with_logits(
                logits.float(), labels, pos_weight=pos_weight,
            )

            preds = (logits > 0).float()
            epoch_correct += (preds == labels).sum().item()
            epoch_total += labels.size(0)

            model.backward(loss)
            model.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % args.gradient_accumulation_steps == 0:
                opt_step += 1

            if step % args.logging_steps == 0:
                avg_loss = epoch_loss / (step + 1)
                acc = epoch_correct / max(epoch_total, 1)
                lr = lr_scheduler.get_last_lr()[0]
                print_rank_0(
                    f"  step {step:>5}/{len(train_loader)}  "
                    f"loss={avg_loss:.4f}  acc={acc:.3f}  lr={lr:.2e}",
                    args.global_rank,
                )
                if wandb and args.global_rank == 0:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/accuracy": acc,
                        "train/lr": lr,
                    }, step=global_step)

            if (val_loader is not None and args.eval_steps > 0
                    and opt_step > 0 and opt_step % args.eval_steps == 0
                    and global_step % args.gradient_accumulation_steps == 0):
                metrics = run_validation(model, prm_head, val_loader, device,
                                         args.global_rank)
                print_rank_0(
                    f"  [val] loss={metrics['val_loss']:.4f} "
                    f"acc={metrics['val_acc']:.3f} "
                    f"(n={metrics['val_n']:,})",
                    args.global_rank,
                )
                if wandb and args.global_rank == 0:
                    wandb.log(metrics, step=global_step)
                if metrics["val_loss"] < best_val_loss:
                    best_val_loss = metrics["val_loss"]
                    _save(model, prm_head, tokenizer, args, global_step,
                          tag="best")

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                _save(model, prm_head, tokenizer, args, global_step)

        avg = epoch_loss / len(train_loader)
        acc = epoch_correct / max(epoch_total, 1)
        print_rank_0(
            f"Epoch {epoch+1} done — train loss: {avg:.4f}, acc: {acc:.3f}",
            args.global_rank,
        )

        if val_loader is not None:
            metrics = run_validation(model, prm_head, val_loader, device,
                                     args.global_rank)
            print_rank_0(
                f"Epoch {epoch+1} val — loss: {metrics['val_loss']:.4f}, "
                f"acc: {metrics['val_acc']:.3f} (n={metrics['val_n']:,})",
                args.global_rank,
            )
            if wandb and args.global_rank == 0:
                wandb.log({
                    **metrics, "epoch": epoch + 1,
                }, step=global_step)
            if metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                _save(model, prm_head, tokenizer, args, global_step, tag="best")

    _save(model, prm_head, tokenizer, args, global_step)

    if wandb and args.global_rank == 0:
        wandb.finish()
    print_rank_0("PRM training complete.", args.global_rank)


def _save(model, prm_head, tokenizer, args, step, tag: str | None = None):
    if args.global_rank != 0:
        return

    save_dir = (
        os.path.join(args.output_dir, tag) if tag else args.output_dir
    )
    os.makedirs(save_dir, exist_ok=True)

    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    head_path = os.path.join(save_dir, "prm_head.pt")
    torch.save(
        {k: v.float().cpu() for k, v in prm_head.state_dict().items()},
        head_path,
    )

    config = {
        "model_name": args.model_name,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "hidden_size": prm_head.score.in_features,
        "backbone": "AutoModel",
        "step": step,
    }
    with open(os.path.join(save_dir, "prm_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    label = f" [{tag}]" if tag else ""
    print_rank_0(
        f"  PRM checkpoint{label} saved to {save_dir} (step {step})",
        args.global_rank,
    )


if __name__ == "__main__":
    main()

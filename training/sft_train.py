#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import hashlib
import random

import numpy as np
import torch
import torch.distributed as dist
import deepspeed
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    get_scheduler,
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.tra import StructuralLoRA, patch_model_with_tra, GraphParser
from training.sft_dataset import (
    GraphInstructSFTDataset,
    GroupedBatchSampler,
    SFTCollator,
)
from training.sft_losses import compute_repr_loss


def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg, flush=True)


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_optimizer_grouped_parameters(model, tra, weight_decay, lr, tra_lr=None):
    no_decay = {"bias", "layer_norm.weight", "layernorm.weight", "norm.weight", "ln_f.weight"}

    tra_param_ids = {id(p) for p in tra.parameters()}

    base_decay = []
    base_no_decay = []
    for n, p in model.named_parameters():
        if not p.requires_grad or id(p) in tra_param_ids:
            continue
        if any(nd in n.lower() for nd in no_decay):
            base_no_decay.append(p)
        else:
            base_decay.append(p)

    groups = [
        {"params": base_decay, "weight_decay": weight_decay, "lr": lr},
        {"params": base_no_decay, "weight_decay": 0.0, "lr": lr},
    ]

    tra_params = [p for p in tra.parameters() if p.requires_grad]
    if tra_params:
        groups.append({
            "params": tra_params,
            "weight_decay": 0.0,
            "lr": tra_lr or lr,
        })

    return [g for g in groups if g["params"]]


def save_checkpoint(model, tra, tokenizer, args, model_name=None):
    zero_stage = args.zero_stage if hasattr(args, "zero_stage") else 0
    model_to_save = model.module if hasattr(model, "module") else model

    if zero_stage == 3:
        state_dict = model._zero3_consolidated_16bit_state_dict()
        if args.global_rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            model_to_save.save_pretrained(args.output_dir, state_dict=state_dict)
    elif args.global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        model_to_save.save_pretrained(args.output_dir)

    if args.global_rank == 0:
        tokenizer.save_pretrained(args.output_dir)

        tra_path = os.path.join(args.output_dir, "tra_weights.pt")
        torch.save(tra.state_dict(), tra_path)

        if model_name:
            with open(os.path.join(args.output_dir, "base_model_name.txt"), "w") as f:
                f.write(model_name)

    print_rank_0(f"Checkpoint saved to {args.output_dir}", args.global_rank)


def get_ds_config(args):
    offload_device = "cpu" if args.offload else "none"
    return {
        "train_micro_batch_size_per_gpu": args.per_device_train_batch_size,
        "train_batch_size": (
            args.per_device_train_batch_size
            * dist.get_world_size()
            * args.gradient_accumulation_steps
        ),
        "steps_per_print": args.logging_steps,
        "zero_optimization": {
            "stage": args.zero_stage,
            "offload_param": {"device": offload_device},
            "offload_optimizer": {"device": offload_device},
            "stage3_param_persistence_threshold": 1e4,
            "stage3_max_live_parameters": 3e7,
            "stage3_prefetch_bucket_size": 3e7,
            "memory_efficient_linear": False,
        },
        "bf16": {"enabled": True},
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }


def parse_args():
    p = argparse.ArgumentParser(description="SFT with compound loss (DeepSpeed)")

    p.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                   choices=["sdpa", "eager", "flash_attention_2"])

    p.add_argument("--data_path", type=str, default="dataset/GraphInstruct-Aug/train.jsonl")
    p.add_argument("--k", type=int, default=4, help="Augmentations per sample")
    p.add_argument("--max_seq_len", type=int, default=2048)

    p.add_argument("--r", type=int, default=8,
                   help="Landmark rank / structural dimension for TRA")
    p.add_argument("--d_max_fp", type=int, default=10,
                   help="Max distance for landmark fingerprint clamping")
    p.add_argument("--tra_layer_start", type=int, default=8)
    p.add_argument("--tra_layer_end", type=int, default=29,
                   help="TRA applied to layers [start, end) (default: 8-28)")

    p.add_argument("--lambda_repr", type=float, default=0.1)
    p.add_argument("--no_mask_prompt", action="store_true",
                   help="Train on full sequence (prompt + response) instead of "
                        "masking prompt tokens. GraphWiz uses full-sequence loss.")

    p.add_argument("--groups_per_batch", type=int, default=4,
                   help="Augmentation groups per micro-batch (controls diversity)")
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--num_train_epochs", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=-1,
                   help="Stop after this many steps (for testing). -1 = full training.")
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--gradient_checkpointing", action="store_true")

    p.add_argument("--zero_stage", type=int, default=2)
    p.add_argument("--offload", action="store_true")
    p.add_argument("--local_rank", type=int, default=-1)

    p.add_argument("--output_dir", type=str, default="checkpoints/sft")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_per_epoch", action="store_true", default=True)

    p.add_argument("--wandb_project", type=str, default="llm-gc-sft")
    p.add_argument("--wandb_run_name", type=str, default="sft-llama3-8b-tra")

    p = deepspeed.add_config_arguments(p)
    return p.parse_args()


def main():
    args = parse_args()

    args.per_device_train_batch_size = args.groups_per_batch * (args.k + 1)

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()

    args.global_rank = dist.get_rank()
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

    print_rank_0("Loading model...", args.global_rank)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    model.resize_token_embeddings(
        int(8 * math.ceil(len(tokenizer) / 8.0))
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print_rank_0(
        f"Full fine-tuning: {trainable:,}/{total:,} params trainable",
        args.global_rank,
    )

    print_rank_0("Setting up TRA...", args.global_rank)
    cfg = model.config
    while hasattr(cfg, "base_model_name_or_path") and not hasattr(cfg, "num_hidden_layers"):
        cfg = getattr(model, "base_model", model).config
        break
    num_layers = getattr(cfg, "num_hidden_layers", 32)
    num_q_heads = getattr(cfg, "num_attention_heads", 32)
    num_kv_heads = getattr(cfg, "num_key_value_heads", 8)

    tra = StructuralLoRA(
        num_layers=num_layers,
        n_q_heads=num_q_heads,
        n_kv_heads=num_kv_heads,
        r=args.r,
        tra_layer_range=(args.tra_layer_start, args.tra_layer_end),
    )
    patch_model_with_tra(model, tra)
    for p in tra.parameters():
        p.requires_grad = True
    print_rank_0(
        f"TRA: {tra.num_extra_parameters:,} params "
        f"(r={args.r}, layers {args.tra_layer_start}-{args.tra_layer_end - 1}, "
        f"Flash Attention-compatible)",
        args.global_rank,
    )

    last_hidden_state = {}
    base = model
    while hasattr(base, "base_model") and base.base_model is not base:
        base = base.base_model
    while hasattr(base, "model") and not hasattr(base, "norm"):
        base = base.model
    base.norm.register_forward_hook(
        lambda mod, inp, out: last_hidden_state.update({"h": out})
    )

    print_rank_0("Loading dataset...", args.global_rank)
    dataset = GraphInstructSFTDataset(args.data_path, k=args.k)
    print_rank_0(
        f"Dataset: {len(dataset):,} samples, "
        f"{dataset.num_groups:,} groups of {dataset.group_size}",
        args.global_rank,
    )

    mask_prompt = not args.no_mask_prompt
    collator = SFTCollator(
        tokenizer=tokenizer,
        r=args.r,
        d_max_fp=args.d_max_fp,
        max_length=args.max_seq_len,
        mask_prompt=mask_prompt,
    )
    print_rank_0(f"  mask_prompt     = {mask_prompt}", args.global_rank)

    sampler = GroupedBatchSampler(
        dataset,
        groups_per_batch=args.groups_per_batch,
        shuffle=True,
        seed=args.seed,
        rank=args.global_rank,
        world_size=dist.get_world_size(),
    )
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
    )

    optimizer_groups = get_optimizer_grouped_parameters(
        model, tra, args.weight_decay, args.learning_rate
    )
    OptimizerClass = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = OptimizerClass(optimizer_groups, lr=args.learning_rate, betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    total_training_steps = args.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_training_steps,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True,
    )

    print_rank_0("***** Running training *****", args.global_rank)
    print_rank_0(f"  Num samples      = {len(dataset):,}", args.global_rank)
    print_rank_0(f"  Num epochs       = {args.num_train_epochs}", args.global_rank)
    print_rank_0(f"  Micro-batch size = {args.per_device_train_batch_size}", args.global_rank)
    print_rank_0(f"  Grad accum steps = {args.gradient_accumulation_steps}", args.global_rank)
    world_size = dist.get_world_size()
    eff_diversity = args.groups_per_batch * world_size * args.gradient_accumulation_steps
    print_rank_0(f"  Groups/batch     = {args.groups_per_batch}", args.global_rank)
    print_rank_0(f"  Batch diversity  = {eff_diversity} unique graphs/update ({world_size} GPUs)", args.global_rank)
    print_rank_0(f"  Total opt steps  = {total_training_steps:,}", args.global_rank)
    print_rank_0(f"  λ_repr           = {args.lambda_repr}", args.global_rank)

    global_step = 0
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"\n{'='*60}\n"
            f"Epoch {epoch + 1}/{args.num_train_epochs}  "
            f"({len(train_dataloader)} micro-batches)\n"
            f"{'='*60}",
            args.global_rank,
        )
        model.train()
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_l_task = 0.0
        epoch_l_repr = 0.0

        for step, batch in enumerate(train_dataloader):
            entity_maps = batch.pop("entity_maps")
            fingerprints = batch.pop("fingerprints")
            repr_positions = batch.pop("repr_positions")
            group_indices = batch.pop("group_indices")

            batch = to_device(batch, device)
            repr_positions = repr_positions.to(device)

            tra.set_graph_data(
                entity_maps.to(device), fingerprints.to(device)
            )

            outputs = model(**batch, use_cache=False)
            l_task = outputs.loss

            l_repr = torch.tensor(0.0, device=device)
            if args.lambda_repr > 0 and "h" in last_hidden_state:
                l_repr = compute_repr_loss(
                    last_hidden_state["h"], repr_positions, group_indices
                )

            loss = l_task + args.lambda_repr * l_repr

            model.backward(loss)
            model.step()

            tra.clear_graph_data()
            last_hidden_state.clear()

            epoch_loss += loss.item()
            epoch_l_task += l_task.item()
            epoch_l_repr += l_repr.item()
            global_step += 1

            if step % args.logging_steps == 0:
                avg_loss = epoch_loss / (step + 1)
                avg_task = epoch_l_task / (step + 1)
                avg_repr = epoch_l_repr / (step + 1)
                lr = lr_scheduler.get_last_lr()[0]
                print_rank_0(
                    f"  step {step:>5}/{len(train_dataloader)}  "
                    f"loss={avg_loss:.4f}  l_task={avg_task:.4f}  "
                    f"l_repr={avg_repr:.4f}  lr={lr:.2e}",
                    args.global_rank,
                )
                if wandb and args.global_rank == 0:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/l_task": l_task.item(),
                        "train/l_repr": l_repr.item(),
                        "train/lr": lr,
                        "train/epoch": epoch + step / len(train_dataloader),
                    }, step=global_step)

            if args.max_steps > 0 and global_step >= args.max_steps:
                print_rank_0(f"  Reached max_steps={args.max_steps}, stopping.", args.global_rank)
                break

        avg = epoch_loss / len(train_dataloader)
        print_rank_0(
            f"Epoch {epoch + 1} done — avg loss: {avg:.4f}",
            args.global_rank,
        )

        if args.save_per_epoch and not (args.max_steps > 0 and global_step >= args.max_steps):
            epoch_dir = os.path.join(args.output_dir, f"epoch_{epoch + 1}")
            save_checkpoint(model, tra, tokenizer, argparse.Namespace(
                output_dir=epoch_dir, global_rank=args.global_rank,
                zero_stage=args.zero_stage,
            ), model_name=args.model_name)

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    save_checkpoint(model, tra, tokenizer, args, model_name=args.model_name)

    if wandb and args.global_rank == 0:
        wandb.finish()

    print_rank_0("Training complete.", args.global_rank)


if __name__ == "__main__":
    main()

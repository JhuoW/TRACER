
from __future__ import annotations

import json
import torch
from torch.utils.data import Dataset, Sampler
from model.tra.graph_parser import GraphParser


ALPACA_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request step by step.\n\n"
    "### Instruction:\n{query}\n\n### Response:"
)


class GraphInstructSFTDataset(Dataset):

    def __init__(self, data_path: str, k: int = 4):
        raw: list[dict] = []
        with open(data_path) as f:
            for line in f:
                raw.append(json.loads(line))

        self.samples = [
            s for s in raw
            if s["augmentation"] == "none" or s["augmentation"].startswith("perm_")
        ]
        self.samples.sort(
            key=lambda s: (s["original_index"], s["augmentation"])
        )

        self.k = k
        self.group_size = k + 1

        n_groups = len(self.samples) // self.group_size
        if len(self.samples) != n_groups * self.group_size:
            self.samples = self.samples[: n_groups * self.group_size]

        self.num_groups = n_groups

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {
            "prompt": ALPACA_PROMPT.format(query=s["query"]),
            "response": s["answer"],
            "query": s["query"],
            "task": s["task"],
            "original_index": s["original_index"],
            "augmentation": s["augmentation"],
        }


class GroupedBatchSampler(Sampler[list[int]]):

    def __init__(
        self,
        dataset: GraphInstructSFTDataset,
        groups_per_batch: int = 4,
        shuffle: bool = True,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.group_size = dataset.group_size
        self.num_groups = dataset.num_groups
        self.groups_per_batch = groups_per_batch
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

    def __iter__(self):
        gs = self.group_size
        group_ids = list(range(self.num_groups))

        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(group_ids), generator=g).tolist()
            group_ids = [group_ids[i] for i in perm]

        if self.world_size > 1:
            remainder = len(group_ids) % self.world_size
            if remainder:
                group_ids += group_ids[:self.world_size - remainder]
            group_ids = group_ids[self.rank::self.world_size]

        batch: list[int] = []
        for gid in group_ids:
            batch.extend(range(gid * gs, (gid + 1) * gs))
            if len(batch) >= self.groups_per_batch * gs:
                yield batch
                batch = []
        if batch:
            yield batch

    def __len__(self) -> int:
        groups_for_rank = (self.num_groups + self.world_size - 1) // self.world_size
        return (groups_for_rank + self.groups_per_batch - 1) // self.groups_per_batch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class SFTCollator:

    def __init__(
        self,
        tokenizer,
        parser: GraphParser | None = None,
        r: int = 8,
        d_max_fp: int = 10,
        max_length: int = 2048,
        mask_prompt: bool = True,
    ):
        self.tokenizer = tokenizer
        self.parser = parser or GraphParser()
        self.r = r
        self.d_max_fp = d_max_fp
        self.max_length = max_length
        self.mask_prompt = mask_prompt

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor | list]:
        prompts = [b["prompt"] for b in batch]
        responses = [b["response"] for b in batch]
        tasks = [b["task"] for b in batch]
        eos = self.tokenizer.eos_token or ""
        full_texts = [p + r + eos for p, r in zip(prompts, responses)]

        enc = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        B, T = input_ids.shape

        labels = input_ids.clone()
        if self.mask_prompt:
            for i, prompt in enumerate(prompts):
                prompt_len = len(
                    self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
                )
                labels[i, :prompt_len] = -100
        labels[attention_mask == 0] = -100

        repr_positions = torch.full((B,), -1, dtype=torch.long)
        for i, prompt in enumerate(prompts):
            prompt_len = len(
                self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
            )
            prompt_len = min(prompt_len, input_ids.shape[1])
            if prompt_len > 0:
                repr_positions[i] = prompt_len - 1

        oi_to_batch: dict[int, list[int]] = {}
        for bi, b in enumerate(batch):
            oi = b["original_index"]
            oi_to_batch.setdefault(oi, []).append(bi)
        group_indices = list(oi_to_batch.values())

        entity_maps = torch.full((B, T), -1, dtype=torch.long)
        max_nodes = 0
        graphs = []
        fp_list = []
        for i in range(B):
            graph = self.parser.parse(full_texts[i], tasks[i])
            graphs.append(graph)
            sp = self.parser.compute_shortest_paths(graph, self.d_max_fp)
            fps, _ = self.parser.compute_landmark_fingerprints(
                graph, sp, r=self.r, d_max_fp=self.d_max_fp,
            )
            fp_list.append(fps)
            if graph["nodes"]:
                max_nodes = max(max_nodes, max(graph["nodes"]) + 1)
            offsets = enc["offset_mapping"][i].tolist()
            emap = self.parser.build_entity_map_from_offsets(
                full_texts[i], offsets, graph
            )
            for pos, nid in emap.items():
                if pos < T:
                    entity_maps[i, pos] = nid

        max_nodes = max(max_nodes, 1)
        fingerprints = torch.zeros(B, max_nodes, self.r, dtype=torch.float32)
        for i in range(B):
            for nid, fp in fp_list[i].items():
                if nid < max_nodes:
                    fingerprints[i, nid] = torch.tensor(fp, dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "entity_maps": entity_maps,
            "fingerprints": fingerprints,
            "repr_positions": repr_positions,
            "group_indices": group_indices,
        }

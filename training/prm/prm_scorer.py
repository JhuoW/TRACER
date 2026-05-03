
from __future__ import annotations

import json
import os
import sys

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from training.prm.step_parser import parse_trace_into_steps
from training.prm.prm_train import PRMHead, PRM_TEMPLATE


class PRMScorer:

    def __init__(
        self,
        checkpoint_dir: str,
        device: str = "cuda:0",
        max_length: int = 2048,
        batch_size: int = 8,
        unparseable_score: float = 0.5,
    ):
        self.device = torch.device(device)
        self.max_length = max_length
        self.batch_size = batch_size
        self.unparseable_score = unparseable_score

        config_path = os.path.join(checkpoint_dir, "prm_config.json")
        with open(config_path) as f:
            self.config = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "left"

        backbone_kind = self.config.get("backbone", "AutoModelForCausalLM")
        loader = AutoModel if backbone_kind == "AutoModel" else AutoModelForCausalLM
        base_model = loader.from_pretrained(
            self.config["model_name"],
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.backbone_kind = backbone_kind

        self.model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        self.model.eval()
        self.model.to(self.device)

        self.head = PRMHead(self.config["hidden_size"]).to(torch.bfloat16)
        head_path = os.path.join(checkpoint_dir, "prm_head.pt")
        state = torch.load(head_path, map_location="cpu")
        state = {k: v.to(torch.bfloat16) for k, v in state.items()}
        self.head.load_state_dict(state)
        self.head.eval()
        self.head.to(self.device)


    def _pooled_hidden(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.backbone_kind == "AutoModel":
            out = self.model(
                input_ids=input_ids, attention_mask=attention_mask,
                use_cache=False,
            )
            last_hidden = out.last_hidden_state
        else:
            out = self.model(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True, use_cache=False,
            )
            last_hidden = out.hidden_states[-1]

        last_pos = attention_mask.sum(dim=1) - 1
        pooled = last_hidden[
            torch.arange(last_hidden.size(0), device=self.device), last_pos,
        ]
        return pooled

    def _score_texts(self, texts: list[str]) -> list[float]:
        scores: list[float] = []
        bs = self.batch_size
        for i in range(0, len(texts), bs):
            chunk = texts[i:i + bs]
            enc = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            pooled = self._pooled_hidden(enc["input_ids"], enc["attention_mask"])
            logits = self.head(pooled.to(self.head.score.weight.dtype))
            probs = torch.sigmoid(logits.float()).detach().cpu().tolist()
            scores.extend(probs)
        return scores


    @staticmethod
    def _render(query: str, prefix: str, step: str) -> str:
        return PRM_TEMPLATE.format(
            query=query,
            prefix=prefix if prefix else "(start of reasoning)",
            step=step,
        )

    @staticmethod
    def _aggregate(step_scores: list[float], mode: str) -> float:
        if not step_scores:
            return 0.0
        if mode == "min":
            return min(step_scores)
        if mode == "mean":
            return sum(step_scores) / len(step_scores)
        if mode == "product":
            r = 1.0
            for s in step_scores:
                r *= s
            return r
        return min(step_scores)

    @torch.no_grad()
    def score_step(self, query: str, prefix: str, step: str) -> float:
        return self._score_texts([self._render(query, prefix, step)])[0]

    @torch.no_grad()
    def score_trace(
        self,
        query: str,
        prediction: str,
        task: str,
        aggregation: str = "min",
    ) -> float:
        steps = parse_trace_into_steps(prediction, task)
        if not steps:
            return self.unparseable_score
        texts = [self._render(query, s["prefix"], s["text"]) for s in steps]
        scores = self._score_texts(texts)
        return self._aggregate(scores, aggregation)

    @torch.no_grad()
    def score_traces_batch(
        self,
        queries: list[str],
        predictions: list[str],
        tasks: list[str],
        aggregation: str = "min",
    ) -> list[float]:
        all_texts: list[str] = []
        ranges: list[tuple[int, int]] = []

        for q, pred, t in zip(queries, predictions, tasks):
            steps = parse_trace_into_steps(pred, t)
            start = len(all_texts)
            for s in steps:
                all_texts.append(self._render(q, s["prefix"], s["text"]))
            ranges.append((start, len(all_texts)))

        if not all_texts:
            return [self.unparseable_score] * len(queries)

        all_scores = self._score_texts(all_texts)

        return [
            self._aggregate(all_scores[a:b], aggregation) if a < b
            else self.unparseable_score
            for a, b in ranges
        ]

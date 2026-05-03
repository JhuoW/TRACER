
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class TRAState:
    entity_map: torch.Tensor | None = None
    fingerprints: torch.Tensor | None = None


class StructuralLoRALayer(nn.Module):

    def __init__(self, n_q_heads: int = 32, n_kv_heads: int = 8, r: int = 8):
        super().__init__()
        self.r = r
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.group_size = n_q_heads // n_kv_heads

        self.W_Q_bias = nn.Parameter(torch.zeros(n_q_heads, r, r))
        self.W_K_bias = nn.Parameter(torch.zeros(n_kv_heads, r, r))

        self.alpha_hat = nn.Parameter(torch.full((n_kv_heads,), -5.0))

    def compute_bias_vectors(
        self,
        fingerprints: torch.Tensor,
        entity_map: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, r = fingerprints.shape
        device = fingerprints.device
        fp = fingerprints.to(dtype=self.W_Q_bias.dtype)

        Z_Q = torch.einsum("bnr,hdr->bnhd", fp, self.W_Q_bias)
        Z_K = torch.einsum("bnr,gdr->bngd", fp, self.W_K_bias)

        safe_ids = entity_map.clamp(min=0)
        entity_mask = (entity_map >= 0).unsqueeze(-1).unsqueeze(-1)

        idx_q = safe_ids.unsqueeze(-1).unsqueeze(-1).expand(
            B, seq_len, self.n_q_heads, r
        )
        bias_Q = Z_Q.gather(1, idx_q.clamp(max=N - 1)) * entity_mask.to(Z_Q.dtype)

        idx_k = safe_ids.unsqueeze(-1).unsqueeze(-1).expand(
            B, seq_len, self.n_kv_heads, r
        )
        bias_K = Z_K.gather(1, idx_k.clamp(max=N - 1)) * entity_mask.to(Z_K.dtype)

        alpha = F.softplus(self.alpha_hat)

        return bias_Q, bias_K, alpha


class StructuralLoRA(nn.Module):

    def __init__(
        self,
        num_layers: int = 32,
        n_q_heads: int = 32,
        n_kv_heads: int = 8,
        r: int = 8,
        tra_layer_range: tuple[int, int] = (8, 29),
    ):
        super().__init__()
        self.num_layers = num_layers
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.r = r
        self.tra_layer_range = tra_layer_range

        self.layers = nn.ModuleDict()
        for l in range(tra_layer_range[0], tra_layer_range[1]):
            self.layers[str(l)] = StructuralLoRALayer(n_q_heads, n_kv_heads, r)

        self.state = TRAState()


    def set_graph_data(
        self,
        entity_map: torch.Tensor,
        fingerprints: torch.Tensor,
    ) -> None:
        self.state.entity_map = entity_map
        self.state.fingerprints = fingerprints

    def clear_graph_data(self) -> None:
        self.state.entity_map = None
        self.state.fingerprints = None


    def is_tra_layer(self, layer_idx: int) -> bool:
        return str(layer_idx) in self.layers

    def get_layer(self, layer_idx: int) -> StructuralLoRALayer:
        return self.layers[str(layer_idx)]


    @property
    def num_extra_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        n = len(self.layers)
        return (
            f"tra_layers={n}, q_heads={self.n_q_heads}, "
            f"kv_heads={self.n_kv_heads}, r={self.r}, "
            f"params={self.num_extra_parameters:,}"
        )

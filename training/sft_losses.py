
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_repr_loss(
    last_hidden_state: torch.Tensor,
    repr_positions: torch.Tensor,
    group_indices: list[list[int]],
) -> torch.Tensor:
    device = last_hidden_state.device
    total_loss = torch.tensor(0.0, device=device)
    num_pairs = 0

    for group in group_indices:
        if len(group) < 2:
            continue

        anchor_idx = group[0]
        anchor_pos = repr_positions[anchor_idx].item()
        if anchor_pos < 0:
            continue
        anchor_h = last_hidden_state[anchor_idx, anchor_pos]

        for other_idx in group[1:]:
            other_pos = repr_positions[other_idx].item()
            if other_pos < 0:
                continue
            other_h = last_hidden_state[other_idx, other_pos]

            cos_sim = F.cosine_similarity(
                anchor_h.unsqueeze(0), other_h.unsqueeze(0)
            )
            total_loss = total_loss + (1.0 - cos_sim.squeeze())
            num_pairs += 1

    if num_pairs == 0:
        return torch.zeros(1, device=device, requires_grad=True).squeeze()

    return total_loss / num_pairs

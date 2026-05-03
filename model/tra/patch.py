
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from functools import wraps
from flash_attn import flash_attn_func

from .attention_bias import StructuralLoRA


def patch_model_with_tra(
    model,
    tra: StructuralLoRA,
) -> StructuralLoRA:
    model.tra = tra

    base = model
    while hasattr(base, "base_model") and base.base_model is not base:
        base = base.base_model
    while hasattr(base, "model") and not hasattr(base, "layers"):
        base = base.model
    layers = base.layers

    for layer_idx, layer in enumerate(layers):
        if not tra.is_tra_layer(layer_idx):
            continue

        attn = layer.self_attn
        _orig_forward = attn.forward

        def _make_forward(orig_fwd, lidx, tra_mod, attn_module):
            tra_layer = tra_mod.get_layer(lidx)

            @wraps(orig_fwd)
            def _forward(hidden_states, attention_mask=None,
                         position_ids=None, **kwargs):
                state = tra_mod.state

                if state.entity_map is None:
                    return orig_fwd(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        **kwargs,
                    )

                device = hidden_states.device
                bsz, q_len, _ = hidden_states.size()

                n_heads = attn_module.config.num_attention_heads
                n_kv_heads = attn_module.config.num_key_value_heads
                head_dim = attn_module.head_dim
                group_size = n_heads // n_kv_heads
                r = tra_layer.r

                query_states = attn_module.q_proj(hidden_states).view(
                    bsz, q_len, n_heads, head_dim
                )
                key_states = attn_module.k_proj(hidden_states).view(
                    bsz, q_len, n_kv_heads, head_dim
                )
                value_states = attn_module.v_proj(hidden_states).view(
                    bsz, q_len, n_kv_heads, head_dim
                )

                position_embeddings = kwargs.get("position_embeddings", None)
                if position_embeddings is not None:
                    cos, sin = position_embeddings
                else:
                    cos, sin = attn_module.rotary_emb(
                        value_states.transpose(1, 2), position_ids
                    )
                from transformers.models.llama.modeling_llama import (
                    apply_rotary_pos_emb,
                )
                q_rot = query_states.transpose(1, 2)
                k_rot = key_states.transpose(1, 2)
                q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
                query_states = q_rot.transpose(1, 2)
                key_states = k_rot.transpose(1, 2)

                emap = state.entity_map.to(device)
                has_entities = (emap >= 0).any()

                if not has_entities:
                    out = flash_attn_func(
                        query_states.to(value_states.dtype),
                        key_states.to(value_states.dtype),
                        value_states,
                        causal=True,
                    )
                    attn_output = out.reshape(bsz, q_len, -1)
                    return attn_module.o_proj(attn_output), None

                if tra_layer.W_Q_bias.device != device:
                    tra_mod.to(device)

                fps = state.fingerprints.to(device)
                bias_Q, bias_K, alpha = tra_layer.compute_bias_vectors(
                    fps, emap, q_len,
                )

                query_states = query_states / math.sqrt(head_dim)

                sqrt_alpha = torch.sqrt(alpha)

                sqrt_alpha_q = sqrt_alpha.repeat_interleave(group_size)
                scaled_bias_Q = bias_Q * sqrt_alpha_q.view(1, 1, -1, 1)

                scaled_bias_K = bias_K * sqrt_alpha.view(1, 1, -1, 1)

                dtype = value_states.dtype
                Q_aug = torch.cat(
                    [query_states.to(dtype), scaled_bias_Q.to(dtype)],
                    dim=-1,
                )
                K_aug = torch.cat(
                    [key_states.to(dtype), scaled_bias_K.to(dtype)],
                    dim=-1,
                )
                V_aug = F.pad(value_states, (0, r))

                out = flash_attn_func(
                    Q_aug, K_aug, V_aug,
                    softmax_scale=1.0,
                    causal=True,
                )

                out = out[:, :, :, :head_dim]

                attn_output = out.reshape(bsz, q_len, -1)
                attn_output = attn_module.o_proj(attn_output)

                return attn_output, None

            return _forward

        attn.forward = _make_forward(_orig_forward, layer_idx, tra, attn)

    return tra

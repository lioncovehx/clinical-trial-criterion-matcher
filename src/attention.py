"""Scaled dot-product attention, written two ways that must give the same answer.

Shapes: q, k, v are [batch, heads, length, head_dim]. Output is [batch, heads, q_len, head_dim].
"""
from contextlib import contextmanager

import torch
import transformers.models.qwen3.modeling_qwen3 as qwen


def causal_allowed(q_start: int, q_end: int, k_start: int, k_end: int, offset: int, device) -> torch.Tensor:
    """True where a query may look at a key. A query may only see its own position and earlier ones.

    offset = k_len - q_len. The queries are the LAST q_len positions, so with a KV cache
    (1 query, T keys) the single query sits at position T-1 and may see every key.
    """
    q_pos = torch.arange(q_start, q_end, device=device)[:, None] + offset
    k_pos = torch.arange(k_start, k_end, device=device)[None, :]
    return k_pos <= q_pos


def _offset(q, k, causal: bool) -> int:
    offset = k.shape[-2] - q.shape[-2]
    if causal and offset < 0:
        raise ValueError("causal attention needs q_len <= k_len, or the first queries would have no keys")
    return offset


def reference_attention(q, k, v, causal: bool = False) -> torch.Tensor:
    """The textbook version: build the whole score table, softmax each row, multiply by V."""
    offset = _offset(q, k, causal)
    scores = q @ k.transpose(-2, -1) * q.shape[-1] ** -0.5
    if causal:
        allowed = causal_allowed(0, q.shape[-2], 0, k.shape[-2], offset, q.device)
        scores = scores.masked_fill(~allowed, -torch.inf)  # exp(-inf) = 0, so hidden keys get no weight
    # Subtract each row's max before exp: the biggest exponent becomes 0, so exp can't overflow.
    scores = scores - scores.max(dim=-1, keepdim=True).values
    weights = scores.exp()
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights @ v


def tiled_attention(q, k, v, block_size: int = 64, causal: bool = False, stats: dict | None = None) -> torch.Tensor:
    """Same result, but only one small tile of the score table exists at a time.

    For each query row we keep three running values while visiting the key blocks left to right:
      m = largest score so far
      l = sum of exp(score - m)                    (the softmax denominator so far)
      o = sum of exp(score - m) * value            (the numerator so far)
    If a new block raises m, the old l and o were measured against the old m, so both are
    multiplied by exp(m_old - m_new). Dividing o by l happens once, at the end.
    """
    q_len, k_len = q.shape[-2], k.shape[-2]
    offset = _offset(q, k, causal)
    scale = q.shape[-1] ** -0.5
    output = q.new_zeros((*q.shape[:-1], v.shape[-1]))

    for q_start in range(0, q_len, block_size):
        q_end = min(q_start + block_size, q_len)
        q_block = q[..., q_start:q_end, :]
        m = torch.full_like(q_block[..., :1], -torch.inf)
        l = torch.zeros_like(m)
        o = q.new_zeros((*q_block.shape[:-1], v.shape[-1]))

        for k_start in range(0, k_len, block_size):
            # Left to right matters: key 0 is visible to every query, so m is finite after the
            # first block, and a later fully hidden row can never compute exp(-inf - -inf) = NaN.
            if causal and k_start > offset + q_end - 1:
                break  # this block and every later one is entirely in the future: skip them
            k_end = min(k_start + block_size, k_len)
            if stats is not None:
                stats["tiles"] = stats.get("tiles", 0) + 1

            scores = q_block @ k[..., k_start:k_end, :].transpose(-2, -1) * scale
            if causal and k_end - 1 > offset + q_start:  # blocks fully in the past need no mask
                scores = scores.masked_fill(~causal_allowed(q_start, q_end, k_start, k_end, offset, q.device), -torch.inf)

            m_new = torch.maximum(m, scores.max(dim=-1, keepdim=True).values)
            rescale = (m - m_new).exp()
            weights = (scores - m_new).exp()
            l = l * rescale + weights.sum(dim=-1, keepdim=True)
            o = o * rescale + weights @ v[..., k_start:k_end, :]
            m = m_new

        output[..., q_start:q_end, :] = o / l
    return output


def tiled_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, block_size=64, **kwargs):
    """Stands in for Qwen3's attention function, so the real model runs on tiled_attention."""
    # Qwen3 has fewer K/V heads than Q heads, so K and V are repeated to match (GQA).
    key, value = qwen.repeat_kv(key, module.num_key_value_groups), qwen.repeat_kv(value, module.num_key_value_groups)
    causal = attention_mask is not None
    if causal:  # prove the model's mask really is the causal rule that tiled_attention applies
        allowed = causal_allowed(0, query.shape[-2], 0, key.shape[-2], key.shape[-2] - query.shape[-2], query.device)
        assert ((attention_mask[0, 0] == 0) == allowed).all(), "the model's mask is not plain causal"
    output = tiled_attention(query, key, value, block_size, causal=causal)
    return output.transpose(1, 2).contiguous(), None


@contextmanager
def tiled_attention_in(model):
    """Inside this block, every layer of a loaded Qwen3 model uses tiled_attention."""
    original_function, original_mode = qwen.eager_attention_forward, model.config._attn_implementation
    qwen.eager_attention_forward, model.config._attn_implementation = tiled_forward, "eager"
    try:
        yield
    finally:
        qwen.eager_attention_forward, model.config._attn_implementation = original_function, original_mode

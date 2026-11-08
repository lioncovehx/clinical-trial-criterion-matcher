"""Greedy generation with and without a KV cache.

Greedy = always take the single highest-scoring next token, so the result is fully repeatable.
"""
import time

import torch

from src.data import load_split, render_prompt


def length_prompts(tok) -> dict[str, torch.Tensor]:
    """Short, medium, and synthetically long prompts, from validation notes."""
    df = load_split("val")
    rendered = sorted((render_prompt(tok, r.note, r.criterion_type, r.criterion_text) for r in df.itertuples()), key=len)
    texts = {"short": rendered[0], "medium": rendered[-1], "long": "\n\n".join(rendered[-1:-4:-1])}
    return {name: tok(text, return_tensors="pt").input_ids.to("cuda") for name, text in texts.items()}


class KVCache:
    """Our own store for each layer's keys and values. It plugs into the model through update().

    Memory for the whole run is reserved up front, and new tokens are written into their slots.
    That avoids copying the whole cache every step, and it makes crop() instant.
    Shape of one layer's K (or V): [batch, kv_heads, capacity, head_dim].
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.keys, self.values, self.lengths = [], [], []

    def update(self, keys, values, layer_idx: int):
        """Called by every attention layer: store the new K/V and hand back all K/V so far."""
        if layer_idx == len(self.keys):  # first call for this layer: reserve the memory
            shape = (*keys.shape[:2], self.capacity, keys.shape[-1])
            self.keys.append(keys.new_zeros(shape))
            self.values.append(values.new_zeros(shape))
            self.lengths.append(0)
        start, end = self.lengths[layer_idx], self.lengths[layer_idx] + keys.shape[-2]
        self.keys[layer_idx][:, :, start:end] = keys
        self.values[layer_idx][:, :, start:end] = values
        self.lengths[layer_idx] = end
        return self.keys[layer_idx][:, :, :end], self.values[layer_idx][:, :, :end]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.lengths[layer_idx] if layer_idx < len(self.lengths) else 0

    def get_query_offset(self, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> tuple[int, int]:
        return self.get_seq_length(layer_idx) + query_length, 0

    def crop(self, length: int) -> None:
        """Forget everything after `length` tokens. Task 5 needs this to discard rejected guesses."""
        self.lengths = [length] * len(self.lengths)

    def memory_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.keys + self.values)


def cached_forward(model, cache: KVCache, tokens, logits_to_keep: int = 1):
    """Run only the NEW tokens through the model. Everything older comes from the cache.

    The position must be the token's true place in the whole sequence, because RoPE rotates
    by absolute position. The mask covers the cached tokens plus the new ones.
    """
    seen, new = cache.get_seq_length(), tokens.shape[1]
    positions = torch.arange(seen, seen + new, device=tokens.device)[None]
    mask = torch.ones(1, seen + new, dtype=torch.long, device=tokens.device)
    return model(input_ids=tokens, attention_mask=mask, position_ids=positions, past_key_values=cache,
                 use_cache=True, logits_to_keep=logits_to_keep).logits


def _clock() -> float:
    torch.cuda.synchronize()  # the GPU runs in the background: wait for it before reading the time
    return time.perf_counter()


def _result(tokens, t0, t1, t2, logits):
    return {"tokens": tokens, "prefill_s": t1 - t0, "decode_s": t2 - t1, "logits": logits}


@torch.inference_mode()
def no_cache_greedy(model, input_ids, max_new_tokens: int, eos_ids=(), keep_logits: bool = False) -> dict:
    """The reference path: reprocess the entire sequence at every step."""
    tokens, kept = [], []
    ids = input_ids
    t0 = _clock()
    t1 = None
    for _ in range(max_new_tokens):
        logits = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1]
        if keep_logits:
            kept.append(logits.float().cpu())
        token = logits.argmax().item()
        tokens.append(token)
        ids = torch.cat([ids, torch.tensor([[token]], device=ids.device)], dim=1)
        if t1 is None:
            t1 = _clock()  # the first pass is the "prefill": it reads the whole prompt
        if token in eos_ids:
            break
    return _result(tokens, t0, t1, _clock(), kept)


@torch.inference_mode()
def kv_cache_greedy(model, input_ids, max_new_tokens: int, eos_ids=(), keep_logits: bool = False) -> dict:
    """The cached path: read the prompt once, then process one new token per step."""
    prompt_len = input_ids.shape[1]
    cache = KVCache(prompt_len + max_new_tokens)
    tokens, kept = [], []

    t0 = _clock()
    logits = cached_forward(model, cache, input_ids)[0, -1]  # prefill
    assert cache.get_seq_length() == prompt_len
    if keep_logits:
        kept.append(logits.float().cpu())
    tokens.append(logits.argmax().item())
    t1 = _clock()

    while len(tokens) < max_new_tokens and tokens[-1] not in eos_ids:
        before = cache.get_seq_length()
        logits = cached_forward(model, cache, torch.tensor([[tokens[-1]]], device=input_ids.device))[0, -1]
        assert cache.get_seq_length() == before + 1, "the cache must grow by exactly one token per step"
        if keep_logits:
            kept.append(logits.float().cpu())
        tokens.append(logits.argmax().item())
    return {**_result(tokens, t0, t1, _clock(), kept), "cache_bytes": cache.memory_bytes()}

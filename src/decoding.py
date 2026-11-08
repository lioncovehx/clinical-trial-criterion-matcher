"""Greedy generation: uncached, KV-cached, and speculative."""
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
    """Per-layer K/V store passed to the model as its cache.

    Preallocated, so adding a token doesn't copy the cache and crop() is O(1).
    K/V shape: [batch, kv_heads, capacity, head_dim].
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.keys, self.values, self.lengths = [], [], []

    def update(self, keys, values, layer_idx: int):
        """Store the new K/V and return everything cached for this layer."""
        if layer_idx == len(self.keys):  # first call for this layer
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
        """Drop everything after `length` tokens."""
        assert all(length <= have for have in self.lengths), "crop can only shorten the cache"
        self.lengths = [length] * len(self.lengths)

    def memory_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.keys + self.values)


def cached_forward(model, cache: KVCache, tokens, logits_to_keep: int = 1):
    """Run only the new tokens; earlier ones come from the cache.

    Positions are absolute because RoPE rotates by position.
    """
    seen, new = cache.get_seq_length(), tokens.shape[1]
    positions = torch.arange(seen, seen + new, device=tokens.device)[None]
    mask = torch.ones(1, seen + new, dtype=torch.long, device=tokens.device)
    return model(input_ids=tokens, attention_mask=mask, position_ids=positions, past_key_values=cache,
                 use_cache=True, logits_to_keep=logits_to_keep).logits


def _clock() -> float:
    torch.cuda.synchronize()  # CUDA is async; flush before reading the clock
    return time.perf_counter()


def _result(tokens, t0, t1, t2, logits):
    return {"tokens": tokens, "prefill_s": t1 - t0, "decode_s": t2 - t1, "logits": logits}


@torch.inference_mode()
def no_cache_greedy(model, input_ids, max_new_tokens: int, eos_ids=(), keep_logits: bool = False) -> dict:
    """Reference path: rerun the whole sequence at every step."""
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
            t1 = _clock()  # first pass is the prefill
        if token in eos_ids:
            break
    return _result(tokens, t0, t1, _clock(), kept)


@torch.inference_mode()
def kv_cache_greedy(model, input_ids, max_new_tokens: int, eos_ids=(), keep_logits: bool = False) -> dict:
    """Prefill once, then process one token per step."""
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


@torch.inference_mode()
def speculative_greedy(target, draft, input_ids, max_new_tokens: int, k: int, eos_ids=()) -> dict:
    """Greedy speculative decoding with a draft and a target model.

    At the start of each round, both caches hold everything except the newest token.
    """
    assert k >= 1 and input_ids.shape[1] >= 2
    device = input_ids.device
    capacity = input_ids.shape[1] + max_new_tokens + k + 2
    target_cache, draft_cache = KVCache(capacity), KVCache(capacity)
    tensor = lambda ids: torch.tensor([ids], device=device)

    t0 = _clock()
    cached_forward(target, target_cache, input_ids[:, :-1])
    cached_forward(draft, draft_cache, input_ids[:, :-1])
    last = input_ids[0, -1].item()
    tokens, accepted_counts, reached = [], [], []
    drafted_total = 0

    while len(tokens) < max_new_tokens and (not tokens or tokens[-1] not in eos_ids):
        # the target's own token adds one more
        guesses = min(k, max_new_tokens - len(tokens) - 1)

        # 1. Draft
        drafted, feed = [], last
        for _ in range(guesses):
            feed = cached_forward(draft, draft_cache, tensor([feed]))[0, -1].argmax().item()
            drafted.append(feed)

        # 2. Verify all guesses in one target pass
        choices = cached_forward(target, target_cache, tensor([last, *drafted]), logits_to_keep=guesses + 1)[0].argmax(-1).tolist()

        # 3. Accept up to the first disagreement
        n = 0
        while n < guesses and drafted[n] == choices[n]:
            n += 1
        # 4. Target's token: a correction, or a bonus if every guess matched
        new = [*drafted[:n], choices[n]]
        accepted_counts.append(n)
        drafted_total += guesses
        reached.extend((token, i < n) for i, token in enumerate(drafted[:n + 1]))

        # 5. Stop at EOS, even inside accepted guesses
        stop = next((i for i, token in enumerate(new) if token in eos_ids), None)
        if stop is not None:
            new = new[:stop + 1]
        tokens.extend(new)
        if stop is not None or len(tokens) >= max_new_tokens:
            break

        # 6. Re-sync both caches
        if n == guesses:  # the draft hasn't processed its last guess
            cached_forward(draft, draft_cache, tensor([drafted[-1]]))
        length = input_ids.shape[1] + len(tokens) - 1
        target_cache.crop(length)
        draft_cache.crop(length)
        last = tokens[-1]

    return {"tokens": tokens, "total_s": _clock() - t0, "accepted_counts": accepted_counts,
            "drafted": drafted_total, "accepted": sum(accepted_counts), "reached": reached,
            "target_passes": len(accepted_counts) + 1}  # + 1 for the prefill

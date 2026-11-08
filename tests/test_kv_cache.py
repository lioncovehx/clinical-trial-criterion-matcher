"""Do cached and uncached greedy decoding produce the same tokens? Run: python -m tests.test_kv_cache

Writes results/kv_cache/equality.csv.
"""
import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.decoding import KVCache, cached_forward, kv_cache_greedy, length_prompts, no_cache_greedy

OUTPUT_LENGTHS = (16, 64, 256)
# bf16 is what we really use. fp32 shows whether a difference is only rounding.
RUNS = [("models/Qwen3-1.7B", torch.bfloat16), ("models/Qwen3-0.6B", torch.float32)]


def load(path, dtype, device="cuda"):
    return AutoTokenizer.from_pretrained(path), AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map=device).eval()


def compare(a, b) -> dict:
    """Identical? If not, where did they first differ, and how close was the model's top choice?"""
    first = next((i for i, (x, y) in enumerate(zip(a["tokens"], b["tokens"])) if x != y), None)
    if first is None and len(a["tokens"]) == len(b["tokens"]):
        return {"identical": True, "first_diff_step": "", "top2_margin_uncached": "", "top2_margin_cached": ""}
    if first is None:
        first = min(len(a["tokens"]), len(b["tokens"]))
    margin = lambda logits: (lambda top: (top[0] - top[1]).item())(logits[first].topk(2).values)
    return {"identical": False, "first_diff_step": first, "top2_margin_uncached": margin(a["logits"]), "top2_margin_cached": margin(b["logits"])}


rows = []
for path, dtype in RUNS:
    tok, model = load(path, dtype)
    for name, ids in length_prompts(tok).items():
        for length in OUTPUT_LENGTHS:
            # EOS is ignored here so every run has exactly the requested length.
            ref = no_cache_greedy(model, ids, length, keep_logits=True)
            cached = kv_cache_greedy(model, ids, length, keep_logits=True)
            result = compare(ref, cached)
            # After the tokens differ, the two runs are reading different text, so only earlier steps are comparable.
            same_steps = result["first_diff_step"] if not result["identical"] else length
            diff = max(abs(ref["logits"][i] - cached["logits"][i]).max().item() for i in range(max(same_steps, 1)))
            row = {"model": path, "dtype": str(dtype), "prompt": name, "prompt_tokens": ids.shape[1], "output_tokens": length,
                   "max_logit_diff_before_divergence": diff, **result}
            rows.append(row)
            print({k: v for k, v in row.items() if k != "model"}, flush=True)

    # Also the real stopping rule: stop at the end-of-turn token, in both paths the same way.
    eos = {tok.eos_token_id}
    ids = next(iter(length_prompts(tok).values()))
    assert no_cache_greedy(model, ids, 64, eos)["tokens"] == kv_cache_greedy(model, ids, 64, eos)["tokens"]

    # crop(): go back to token 10 and continue. It must match the run that never went back.
    ids = next(iter(length_prompts(tok).values()))
    full = kv_cache_greedy(model, ids, 32)["tokens"]
    cache = KVCache(ids.shape[1] + 64)
    with torch.inference_mode():
        cached_forward(model, cache, ids)
        for token in full[:20]:
            cached_forward(model, cache, torch.tensor([[token]], device="cuda"))  # run past the crop point
        cache.crop(ids.shape[1] + 10)
        assert cache.get_seq_length() == ids.shape[1] + 10
        after = []
        token = full[10]
        for _ in range(21):
            logits = cached_forward(model, cache, torch.tensor([[token]], device="cuda"))[0, -1]
            token = logits.argmax().item()
            after.append(token)
    assert full[11:32] == after, "crop then continue did not match the uninterrupted run"
    print(path, "EOS stopping and crop() checks passed", flush=True)
    del model
    torch.cuda.empty_cache()

out = Path("results/kv_cache/equality.csv")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

# Any bf16 difference is rerun on the SAME model and prompt in fp32, where rounding almost disappears.
# Qwen3-1.7B in fp32 needs about 7 GB and doesn't fit on the GPU with room to run, so this uses the CPU.
# It only generates a little past the step that differed, which keeps the slow CPU run short.
bf16 = [r for r in rows if r["dtype"] == "torch.bfloat16"]
cases = {(r["prompt"], r["first_diff_step"]) for r in bf16 if not r["identical"]}
rerun = []
if cases:
    tok, model = load(RUNS[0][0], torch.float32, "cpu")
    prompts = length_prompts(tok)
    for prompt, step in sorted(cases):
        ids = prompts[prompt].to("cpu")
        tokens = step + 3
        same = no_cache_greedy(model, ids, tokens)["tokens"] == kv_cache_greedy(model, ids, tokens)["tokens"]
        rerun.append({"prompt": prompt, "bf16_first_diff_step": step, "tokens_generated": tokens, "identical_in_fp32": same})
        print("fp32 rerun of a bf16 difference:", rerun[-1], flush=True)
    del model

(out.parent / "fp32_rerun.json").write_text(json.dumps(rerun, indent=2))
fp32 = [r for r in rows if r["dtype"] == "torch.float32"]
print(f"bf16: {sum(r['identical'] for r in bf16)}/{len(bf16)} identical | fp32: {sum(r['identical'] for r in fp32)}/{len(fp32)} identical")
assert all(r["identical"] for r in fp32) and all(r["identical_in_fp32"] for r in rerun), "fp32 runs differ: this would be a real bug"
print("all KV cache tests passed")

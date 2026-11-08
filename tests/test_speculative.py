"""Does speculative decoding produce exactly what the target alone would? Run: python -m tests.test_speculative

Writes results/speculative/tokenizer_check.json and equality.csv.
"""
import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import load_split, render_prompt
from src.decoding import kv_cache_greedy, length_prompts, speculative_greedy
from src.pipeline import load_model

TARGET, DRAFT, ADAPTER = "models/Qwen3-1.7B", "models/Qwen3-0.6B", "checkpoints/lora_v2_lowlr/best"
KS = (1, 2, 4, 6, 8)
out = Path("results/speculative")
out.mkdir(parents=True, exist_ok=True)

# 1. Tokenizers must match
tok_t, tok_d = AutoTokenizer.from_pretrained(TARGET), AutoTokenizer.from_pretrained(DRAFT)
row = load_split("val").iloc[0]
check = {"same_vocab_size": len(tok_t) == len(tok_d), "same_vocab": tok_t.get_vocab() == tok_d.get_vocab(),
         "same_special_tokens": tok_t.special_tokens_map == tok_d.special_tokens_map,
         "same_eos": tok_t.eos_token_id == tok_d.eos_token_id,
         "same_rendered_prompt": render_prompt(tok_t, row.note, row.criterion_type, row.criterion_text)
         == render_prompt(tok_d, row.note, row.criterion_type, row.criterion_text)}
(out / "tokenizer_check.json").write_text(json.dumps(check, indent=2))
print("tokenizer check:", check)
assert all(check.values()), "the tokenizers do not match"


def prompts_for(tok) -> list[tuple[str, torch.Tensor, int, set]]:
    """Real pipeline questions (stop at EOS) and long free-running prompts (fixed length)."""
    rows = load_split("val").head(10)
    eos = {tok.eos_token_id}
    cases = [(f"val_{r.annotation_id}", tok(render_prompt(tok, r.note, r.criterion_type, r.criterion_text), return_tensors="pt").input_ids.to("cuda"), 64, eos)
             for r in rows.itertuples()]
    cases += [(f"{name}_128", ids, 128, set()) for name, ids in length_prompts(tok).items()]
    return cases


def compare_all(target, draft, label: str) -> list[dict]:
    rows = []
    for name, ids, length, eos in prompts_for(tok_t):
        plain = kv_cache_greedy(target, ids, length, eos, keep_logits=True)
        for k in KS:
            spec = speculative_greedy(target, draft, ids, length, k, eos)
            first = next((i for i, (a, b) in enumerate(zip(plain["tokens"], spec["tokens"])) if a != b), None)
            same = first is None and len(plain["tokens"]) == len(spec["tokens"])
            margin = ""
            if first is not None:
                top = plain["logits"][first].topk(2).values
                margin = (top[0] - top[1]).item()
            rows.append({"setup": label, "prompt": name, "k": k, "tokens": len(plain["tokens"]), "identical": same, "first_diff_step": "" if first is None else first,
                         "plain_top2_margin": margin, "accept_rate": round(spec["accepted"] / max(spec["drafted"], 1), 3)})
            if not same:
                print("difference:", rows[-1], flush=True)
    good = sum(r["identical"] for r in rows)
    print(f"{label}: {good}/{len(rows)} identical", flush=True)
    return rows


# 2. Real setup: fine-tuned target (bf16), plain draft
_, target = load_model(TARGET, ADAPTER)
draft = AutoModelForCausalLM.from_pretrained(DRAFT, dtype=torch.bfloat16, device_map="cuda").eval()
results = compare_all(target, draft, "real_bf16")
del target, draft
torch.cuda.empty_cache()

# 3. fp32 logic check. 1.7B doesn't fit in fp32 beside a second model, so 0.6B (fp32) is the
#    target and 1.7B (bf16) the draft. Only token IDs pass between them.
target = AutoModelForCausalLM.from_pretrained(DRAFT, dtype=torch.float32, device_map="cuda").eval()
draft = AutoModelForCausalLM.from_pretrained(TARGET, dtype=torch.bfloat16, device_map="cuda").eval()
proof = compare_all(target, draft, "logic_proof_fp32")

with open(out / "equality.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, results[0].keys())
    writer.writeheader()
    writer.writerows(results + proof)

assert all(r["identical"] for r in proof), "fp32 runs differ: this would be a real bug"
print("all speculative decoding tests passed")

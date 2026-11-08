"""Measure speculative decoding against plain cached greedy, for several draft lengths.

Run: python -m scripts.bench_speculative
Writes results/speculative/benchmark.csv, accept_by_type.csv and benchmark.png. Close other GPU programs first.
"""
import argparse
import collections
import csv
import statistics
import time
from pathlib import Path

import matplotlib
import torch
from transformers import AutoModelForCausalLM

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import load_split, render_prompt
from src.decoding import KVCache, cached_forward, kv_cache_greedy, length_prompts, speculative_greedy
from src.pipeline import load_model

parser = argparse.ArgumentParser()
parser.add_argument("--drafts", nargs="+", default=["base", "tuned"], choices=["base", "tuned"])
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--rows", type=int, default=20)
args = parser.parse_args()

TARGET, DRAFT = "models/Qwen3-1.7B", "models/Qwen3-0.6B"
ADAPTERS = {"base": None, "tuned": "checkpoints/lora_draft/best"}
KS = (1, 2, 4, 6, 8)
tok, target = load_model(TARGET, "checkpoints/lora_v2_lowlr/best")
EOS = {tok.eos_token_id}
val = load_split("val").head(args.rows)
WORKLOADS = {
    "pipeline": [(tok(render_prompt(tok, r.note, r.criterion_type, r.criterion_text), return_tensors="pt").input_ids.to("cuda"), 64, EOS) for r in val.itertuples()],
    "long_free_text": [(ids, 128, set()) for ids in length_prompts(tok).values()],
}


def step_ms(model, ids, steps=32) -> float:
    """Average time of one cached decode step, in ms."""
    cache = KVCache(ids.shape[1] + steps + 2)
    with torch.inference_mode():
        token = cached_forward(model, cache, ids)[0, -1].argmax().item()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            token = cached_forward(model, cache, torch.tensor([[token]], device="cuda"))[0, -1].argmax().item()
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / steps * 1000


def kind(token: int) -> str:
    text = tok.decode([token]).strip()
    if text.isdigit():
        return "sentence ID"
    return "label word" if text.isalpha() and text not in ("evidence", "label") else "format"


rows, kinds = [], []
for draft_name in args.drafts:
    _, draft = load_model(DRAFT, ADAPTERS[draft_name])
    ids = WORKLOADS["pipeline"][0][0]
    step_ms(target, ids), step_ms(draft, ids)  # warm-up
    target_ms, draft_ms = statistics.median(step_ms(target, ids) for _ in range(5)), statistics.median(step_ms(draft, ids) for _ in range(5))
    ratio = draft_ms / target_ms
    print(f"draft={draft_name}: target step {target_ms:.2f} ms, draft step {draft_ms:.2f} ms, cost ratio {ratio:.2f}", flush=True)

    for workload, cases in WORKLOADS.items():
        torch.cuda.reset_peak_memory_stats()
        plain_runs = [[kv_cache_greedy(target, x, n, e) for x, n, e in cases] for _ in range(args.repeats)]
        plain = statistics.median(sum(r["prefill_s"] + r["decode_s"] for r in run) for run in plain_runs)
        n_tokens = sum(len(r["tokens"]) for r in plain_runs[0])
        rows.append({"draft": draft_name, "workload": workload, "k": "plain", "total_ms": round(plain * 1000, 1), "tokens_per_s": round(n_tokens / plain, 1), "speedup": 1.0,
                     "accept_rate": "", "mean_accepted_per_round": "", "target_passes_per_token": "", "alpha": "", "predicted_speedup": "", "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "accepted_hist": ""})
        print(rows[-1], flush=True)

        for k in KS:
            torch.cuda.reset_peak_memory_stats()
            times, last = [], None
            for _ in range(args.repeats):
                results = [speculative_greedy(target, draft, x, n, k, e) for x, n, e in cases]
                times.append(sum(r["total_s"] for r in results))
                last = results
            total = statistics.median(times)
            tokens = sum(len(r["tokens"]) for r in last)
            drafted, accepted = sum(r["drafted"] for r in last), sum(r["accepted"] for r in last)
            counts = [n for r in last for n in r["accepted_counts"]]
            # alpha: acceptance rate among guesses actually reached
            reached = sum(len(r["reached"]) for r in last)
            alpha = accepted / reached if reached else 0.0
            expected = sum(alpha ** i for i in range(k + 1))  # expected tokens per round (Leviathan et al.)
            rows.append({"draft": draft_name, "workload": workload, "k": k, "total_ms": round(total * 1000, 1), "tokens_per_s": round(tokens / total, 1), "speedup": round(plain / total, 3),
                         "accept_rate": round(accepted / max(drafted, 1), 3), "mean_accepted_per_round": round(statistics.mean(counts), 2),
                         "target_passes_per_token": round(sum(r["target_passes"] for r in last) / tokens, 3), "alpha": round(alpha, 3),
                         "predicted_speedup": round(expected / (1 + k * ratio), 3), "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                         "accepted_hist": dict(sorted(collections.Counter(counts).items()))})
            print(rows[-1], flush=True)
            if workload == "pipeline":
                tally = collections.defaultdict(lambda: [0, 0])
                for r in last:
                    for token, ok in r["reached"]:
                        tally[kind(token)][0] += ok
                        tally[kind(token)][1] += 1
                kinds += [{"draft": draft_name, "k": k, "token_kind": name, "accepted": a, "reached": b, "rate": round(a / b, 3)} for name, (a, b) in tally.items()]
    del draft
    torch.cuda.empty_cache()

out = Path("results/speculative")
out.mkdir(parents=True, exist_ok=True)
for name, data in (("benchmark.csv", rows), ("accept_by_type.csv", kinds)):
    with open(out / name, "w", newline="") as f:
        writer = csv.DictWriter(f, data[0].keys())
        writer.writeheader()
        writer.writerows(data)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for draft_name in args.drafts:
    for workload in WORKLOADS:
        group = [r for r in rows if r["draft"] == draft_name and r["workload"] == workload and r["k"] != "plain"]
        for ax, key in zip(axes, ("speedup", "accept_rate", "predicted_speedup")):
            ax.plot([r["k"] for r in group], [r[key] for r in group], marker="o", label=f"{draft_name} / {workload}")
for ax, key in zip(axes, ("measured speedup", "acceptance rate", "predicted speedup")):
    ax.set_xlabel("k (tokens drafted)"); ax.set_ylabel(key); ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(out / "benchmark.png", dpi=150)
print("wrote", out / "benchmark.csv")

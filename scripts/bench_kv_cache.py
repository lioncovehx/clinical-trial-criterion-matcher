"""Measure cached vs uncached greedy decoding. Close other GPU programs first.

Run: python -m scripts.bench_kv_cache
Writes results/kv_cache/benchmark.csv and benchmark.png.
"""
import argparse
import csv
import statistics
from pathlib import Path

import matplotlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.decoding import kv_cache_greedy, length_prompts, no_cache_greedy

parser = argparse.ArgumentParser()
parser.add_argument("--outputs", type=int, nargs="+", default=[16, 64, 256])
parser.add_argument("--repeats", type=int, default=5)
args = parser.parse_args()

MODEL = "models/Qwen3-1.7B"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
cfg = model.config
BYTES_PER_TOKEN = cfg.num_hidden_layers * cfg.num_key_value_heads * (cfg.hidden_size // cfg.num_attention_heads) * 2 * 2  # K and V, 2 bytes each
print("KV cache formula:", BYTES_PER_TOKEN, "bytes per token")

prompts = length_prompts(tok)
kv_cache_greedy(model, prompts["short"], 8)  # warm-up
no_cache_greedy(model, prompts["short"], 8)

rows = []
for prompt_name, ids in prompts.items():
    for length in args.outputs:
        for method, fn in (("no_cache", no_cache_greedy), ("kv_cache", kv_cache_greedy)):
            runs, peaks = [], []
            for _ in range(args.repeats):
                base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                runs.append(fn(model, ids, length))  # no EOS: fixed output length
                peaks.append((torch.cuda.max_memory_allocated() - base) / 1e9)
            prefill, decode = [r["prefill_s"] * 1000 for r in runs], [r["decode_s"] * 1000 for r in runs]
            row = {"prompt": prompt_name, "prompt_tokens": ids.shape[1], "output_tokens": length, "method": method,
                   "prefill_ms": round(statistics.median(prefill), 2),
                   "decode_ms": round(statistics.median(decode), 2),
                   "total_ms": round(statistics.median(p + d for p, d in zip(prefill, decode)), 2),
                   # prefill produces the first token
                   "decode_tokens_per_s": round((length - 1) / (statistics.median(decode) / 1000), 1),
                   "peak_extra_gb": round(max(peaks), 4),
                   "cache_formula_gb": round((ids.shape[1] + length) * BYTES_PER_TOKEN / 1e9, 4) if method == "kv_cache" else "",
                   "cache_measured_gb": round(runs[0]["cache_bytes"] / 1e9, 4) if method == "kv_cache" else ""}
            rows.append(row)
            print(row, flush=True)

out = Path("results/kv_cache")
out.mkdir(parents=True, exist_ok=True)
with open(out / "benchmark.csv", "w", newline="") as f:
    f.write(f"# gpu={torch.cuda.get_device_name(0)} model={MODEL} dtype=bf16 torch={torch.__version__} batch=1 median_of={args.repeats}\n")
    writer = csv.DictWriter(f, rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for prompt_name in prompts:
    for method, style in (("no_cache", "--"), ("kv_cache", "-")):
        group = [r for r in rows if r["prompt"] == prompt_name and r["method"] == method]
        x = [r["output_tokens"] for r in group]
        axes[0].plot(x, [r["total_ms"] for r in group], style, marker="o", label=f"{prompt_name} {method}")
        axes[1].plot(x, [r["decode_tokens_per_s"] for r in group], style, marker="o", label=f"{prompt_name} {method}")
        axes[2].plot([r["prompt_tokens"] + r["output_tokens"] for r in group], [r["peak_extra_gb"] for r in group], style, marker="o", label=f"{prompt_name} {method}")
for ax, xlabel, ylabel in zip(axes, ("output tokens", "output tokens", "total sequence length"), ("total latency (ms)", "decode tokens/s", "peak extra memory (GB)")):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.legend(fontsize=6)
axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[1].set_xscale("log")
fig.tight_layout()
fig.savefig(out / "benchmark.png", dpi=150)
print("wrote", out / "benchmark.csv")

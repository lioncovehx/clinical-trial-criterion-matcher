"""Compare speed and peak memory: reference vs tiled vs PyTorch's fused attention.

Run: python -m scripts.bench_attention
Writes results/attention/benchmark.csv and benchmark.png. Close other GPU programs first.
"""
import argparse
import csv
import statistics
import time
from pathlib import Path

import matplotlib
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.attention import causal_allowed, reference_attention, tiled_attention

parser = argparse.ArgumentParser()
parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096, 8192])
args = parser.parse_args()

HEADS, HEAD_DIM, BLOCK, REPEATS = 8, 128, 64, 5
DEVICE, DTYPE = "cuda", torch.float16  # the FlashAttention kernels want fp16 or bf16


def fused(backend):
    def run(q, k, v, mask):
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return run


METHODS = {
    "reference": lambda q, k, v, mask: reference_attention(q, k, v, causal=True),
    f"tiled_{BLOCK}": lambda q, k, v, mask: tiled_attention(q, k, v, BLOCK, causal=True),
    "sdpa_math": fused(SDPBackend.MATH),
    "sdpa_efficient": fused(SDPBackend.EFFICIENT_ATTENTION),
    "sdpa_flash": fused(SDPBackend.FLASH_ATTENTION),
    "sdpa_cudnn": fused(SDPBackend.CUDNN_ATTENTION),
}


def measure(fn, q, k, v, mask):
    fn(q, k, v, mask)  # warm-up: the first run includes one-time setup costs
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(REPEATS):
        torch.cuda.synchronize()  # the GPU works in the background: wait before starting the clock
        start = time.perf_counter()
        out = fn(q, k, v, mask)
        torch.cuda.synchronize()  # ...and wait for it to finish before stopping the clock
        times.append(time.perf_counter() - start)
        del out
    return times, (torch.cuda.max_memory_allocated() - base) / 1e9


torch.manual_seed(0)
rows = []
with torch.inference_mode():
    for n in args.sizes:
        q, k, v = (torch.randn(1, HEADS, n, HEAD_DIM, device=DEVICE, dtype=DTYPE) for _ in range(3))
        for name, fn in METHODS.items():
            try:
                times, peak = measure(fn, q, k, v, None)
            except (RuntimeError, torch.OutOfMemoryError) as e:
                print(f"N={n:5} {name:15} not available: {str(e).splitlines()[0][:80]}")
                torch.cuda.empty_cache()
                continue
            row = {"n": n, "method": name, "median_ms": round(statistics.median(times) * 1000, 3),
                   "std_ms": round(statistics.stdev(times) * 1000, 3), "peak_gb": round(peak, 4)}
            rows.append(row)
            print(row, flush=True)
        del q, k, v
        torch.cuda.empty_cache()

out = Path("results/attention")
out.mkdir(parents=True, exist_ok=True)
info = {"gpu": torch.cuda.get_device_name(0), "dtype": str(DTYPE), "heads": HEADS, "head_dim": HEAD_DIM, "torch": torch.__version__}
with open(out / "benchmark.csv", "w", newline="") as f:
    f.write(f"# {info}\n")
    writer = csv.DictWriter(f, rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for name in {r["method"]: 0 for r in rows}:
    group = [r for r in rows if r["method"] == name]
    axes[0].plot([r["n"] for r in group], [r["median_ms"] for r in group], marker="o", label=name)
    axes[1].plot([r["n"] for r in group], [max(r["peak_gb"], 1e-4) for r in group], marker="o", label=name)
for ax, label in zip(axes, ("time (ms)", "peak extra memory (GB)")):
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("sequence length N"); ax.set_ylabel(label); ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(out / "benchmark.png", dpi=150)
print("wrote", out / "benchmark.csv")

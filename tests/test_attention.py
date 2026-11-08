"""Do reference and tiled attention agree? Run: python -m tests.test_attention

Writes one line per configuration to results/attention/errors.csv.
"""
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from src.attention import causal_allowed, reference_attention, tiled_attention

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Max absolute error by (dtype, Q scale): a few times the worst measured.
TOLERANCE = {("fp32", 1): 1e-5, ("fp32", 100): 1e-3, ("fp16", 1): 1e-2, ("fp16", 100): 2e-2,
             ("bf16", 1): 5e-2, ("bf16", 100): 1e-1}
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


# 1. Chunked online softmax equals softmax
scores = [1.0, -2.0, 0.5, 7.0, 3.0, -1.0, 9.0, 0.0]
exact = [math.exp(s) / sum(math.exp(x) for x in scores) for s in scores]
values = list(range(len(scores)))
m, l, o = -math.inf, 0.0, 0.0
for i in range(0, len(scores), 3):
    chunk, chunk_values = scores[i:i + 3], values[i:i + 3]
    m_new = max(m, *chunk)
    rescale = math.exp(m - m_new) if m != -math.inf else 0.0
    l = l * rescale + sum(math.exp(s - m_new) for s in chunk)
    o = o * rescale + sum(math.exp(s - m_new) * v for s, v in zip(chunk, chunk_values))
    m = m_new
assert math.isclose(o / l, sum(p * v for p, v in zip(exact, values)), rel_tol=1e-12)

# 2. Causal mask, small cases
assert causal_allowed(0, 3, 0, 3, 0, "cpu").tolist() == [[True, False, False], [True, True, False], [True, True, True]]
assert causal_allowed(0, 1, 0, 3, 2, "cpu").tolist() == [[True, True, True]]  # 1 query, 3 keys: sees everything
assert causal_allowed(0, 2, 0, 4, 2, "cpu").tolist() == [[True, True, True, False], [True, True, True, True]]

# 3. Future tiles are skipped: 36 of 64 visited
x = torch.randn(1, 1, 512, 64, device=DEVICE)
stats = {}
tiled_attention(x, x, x, block_size=64, causal=True, stats=stats)
assert stats["tiles"] == 36, stats

# 4. Reference vs tiled vs PyTorch over a grid
torch.manual_seed(0)
rows = []
for name, dtype in DTYPES.items():
    for head_dim in (64, 128):
        for k_len in (1, 7, 64, 127, 513):
            for q_len in sorted({k_len, max(1, k_len // 2), 1}):
                for causal in (False, True):
                    for block in (16, 32, 64, 128):
                        for q_scale in (1, 100):  # 100: very large scores
                            q = torch.randn(1, 2, q_len, head_dim, device=DEVICE, dtype=dtype) * q_scale
                            k = torch.randn(1, 2, k_len, head_dim, device=DEVICE, dtype=dtype)
                            v = torch.randn(1, 2, k_len, head_dim, device=DEVICE, dtype=dtype)
                            ref = reference_attention(q, k, v, causal)
                            tiled = tiled_attention(q, k, v, block, causal)
                            mask = causal_allowed(0, q_len, 0, k_len, k_len - q_len, DEVICE) if causal else None
                            sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                            assert ref.isfinite().all() and tiled.isfinite().all(), (name, k_len, q_len, causal, block, q_scale)
                            error = (ref.float() - tiled.float()).abs()
                            rows.append({"dtype": name, "head_dim": head_dim, "k_len": k_len, "q_len": q_len, "causal": causal,
                                         "block": block, "q_scale": q_scale, "max_err": error.max().item(),
                                         "mean_err": error.mean().item(),
                                         "max_err_vs_sdpa": (ref.float() - sdpa.float()).abs().max().item()})

out = Path("results/attention/errors.csv")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"{'dtype':6} {'scale':>5} {'cases':>6} {'worst tiled err':>16} {'mean tiled err':>15} {'worst SDPA err':>15} {'limit':>8}")
for name in DTYPES:
    for scale in (1, 100):
        group = [r for r in rows if r["dtype"] == name and r["q_scale"] == scale]
        worst = max(r["max_err"] for r in group)
        print(f"{name:6} {scale:5} {len(group):6} {worst:16.2e} {sum(r['mean_err'] for r in group) / len(group):15.2e} "
              f"{max(r['max_err_vs_sdpa'] for r in group):15.2e} {TOLERANCE[name, scale]:8.0e}")
        assert worst <= TOLERANCE[name, scale], (name, scale, worst)
print("all attention tests passed")

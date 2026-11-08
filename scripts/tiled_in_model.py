"""Swap your tiled attention into every layer of Qwen3 and check nothing changes.

Run: python -m scripts.tiled_in_model
"""
import json
from pathlib import Path

import torch
import transformers.models.qwen3.modeling_qwen3 as qwen
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.attention import tiled_forward
from src.data import load_split, render_prompt

# bf16 is what we really use. fp32 is the proof: with 4-byte numbers rounding almost disappears,
# so any difference that remains would be a real bug. (The 0.6B model has the same design and fits in fp32.)
RUNS = [("models/Qwen3-1.7B", torch.bfloat16), ("models/Qwen3-0.6B", torch.float32)]
original = qwen.eager_attention_forward


def run(model, inputs):
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False).logits
        tokens = model.generate(**inputs, max_new_tokens=48, do_sample=False, temperature=None, top_p=None, top_k=None)
    return logits.float(), tokens[0].tolist()


results = []
df = load_split("val").head(5)
for path, dtype in RUNS:
    tok = AutoTokenizer.from_pretrained(path)
    # "eager" makes Qwen3 call the plain Python attention function, which is the one we replace.
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map="cuda", attn_implementation="eager").eval()
    for r in df.itertuples():
        inputs = tok(render_prompt(tok, r.note, r.criterion_type, r.criterion_text), return_tensors="pt").to("cuda")
        qwen.eager_attention_forward = original
        base_logits, base_tokens = run(model, inputs)
        qwen.eager_attention_forward = tiled_forward
        tiled_logits, tiled_tokens = run(model, inputs)
        qwen.eager_attention_forward = original
        row = {"model": path, "dtype": str(dtype), "annotation_id": int(r.annotation_id), "prompt_tokens": inputs.input_ids.shape[1],
               "max_logit_diff": (base_logits - tiled_logits).abs().max().item(),
               "prediction_agreement": (base_logits.argmax(-1) == tiled_logits.argmax(-1)).float().mean().item(),
               "tokens_identical": base_tokens == tiled_tokens}
        results.append(row)
        print(row, flush=True)
    del model
    torch.cuda.empty_cache()

Path("results/attention").mkdir(parents=True, exist_ok=True)
Path("results/attention/in_model.json").write_text(json.dumps(results, indent=2))
assert all(r["tokens_identical"] for r in results), "the generated tokens differ"
assert all(r["max_logit_diff"] < 1e-2 for r in results if r["dtype"] == "torch.float32"), "fp32 logits differ: a real bug"
print("tiled attention inside Qwen3: identical tokens on every prompt; fp32 logits match")

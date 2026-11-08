"""Check that Qwen3-1.7B loads on the GPU and gives a non-thinking reply.

Run from the repo root:  python scripts/smoke_test.py
"""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "models/Qwen3-1.7B"  # or "Qwen/Qwen3-1.7B" from the Hub

torch.manual_seed(0)

# bf16: about 2 bytes per parameter
t0 = time.perf_counter()
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cuda")
model.eval()
torch.cuda.synchronize()  # CUDA is async; flush before timing
print(f"load time: {time.perf_counter() - t0:.1f}s")
print(f"memory after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# enable_thinking=False prefills an empty <think> block.
messages = [{"role": "user", "content": "In one sentence: what is an exclusion criterion in a clinical trial?"}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
print("---- prompt string the model actually sees ----")
print(prompt)
inputs = tok(prompt, return_tensors="pt").to("cuda")
print(f"prompt tokens: {inputs.input_ids.shape[1]}")

torch.cuda.reset_peak_memory_stats()
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
new_tokens = out[0, inputs.input_ids.shape[1]:]  # reply only
reply = tok.decode(new_tokens, skip_special_tokens=True)
print("---- reply ----")
print(reply)
print(f"generated tokens: {len(new_tokens)}")

# weights + KV cache + activations
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"peak memory during generate: {peak:.2f} GB")

assert "<think>" not in reply, "thinking mode leaked into the output"
assert peak < 8, "used more than the 8 GB card has"
print("SMOKE TEST PASSED")

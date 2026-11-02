"""Smoke test: can we load Qwen3-1.7B on the GPU and get a sane, non-thinking reply?

Run from the repo root:  python scripts/smoke_test.py
"""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "models/Qwen3-1.7B"  # local folder; on a fresh clone use "Qwen/Qwen3-1.7B"

torch.manual_seed(0)  # greedy decoding is deterministic anyway, but log a seed by habit

# 1. Load. bf16 = 2 bytes per weight, so expect roughly params x 2 bytes of GPU memory.
t0 = time.perf_counter()
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cuda")
model.eval()  # inference mode: disables dropout
torch.cuda.synchronize()  # GPU work is async; wait before reading the clock
print(f"load time: {time.perf_counter() - t0:.1f}s")
print(f"memory after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# 2. Prompt. The chat template wraps the message in Qwen's special role tokens.
#    enable_thinking=False makes the template pre-fill an EMPTY <think></think> block,
#    so the model skips reasoning and answers directly.
messages = [{"role": "user", "content": "In one sentence: what is an exclusion criterion in a clinical trial?"}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
print("---- prompt string the model actually sees ----")
print(prompt)
inputs = tok(prompt, return_tensors="pt").to("cuda")
print(f"prompt tokens: {inputs.input_ids.shape[1]}")

# 3. Generate greedily (do_sample=False = always pick the highest-scoring token).
torch.cuda.reset_peak_memory_stats()
with torch.inference_mode():  # no gradients -> far less memory
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
new_tokens = out[0, inputs.input_ids.shape[1]:]  # drop the prompt, keep only the reply
reply = tok.decode(new_tokens, skip_special_tokens=True)
print("---- reply ----")
print(reply)
print(f"generated tokens: {len(new_tokens)}")

# 4. Memory. Peak includes weights + KV cache + temporary activations.
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"peak memory during generate: {peak:.2f} GB")

assert "<think>" not in reply, "thinking mode leaked into the output"
assert peak < 8, "used more than the 8 GB card has"
print("SMOKE TEST PASSED")

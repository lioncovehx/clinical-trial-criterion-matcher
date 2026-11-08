"""Fine-tune with LoRA on the train split, using settings from a config file.

Run:          python -m scripts.train_lora --config configs/lora_v1.json
Overfit check: python -m scripts.train_lora --config configs/lora_v1.json --name overfit --limit 8 --epochs 25

Keeps the adapter with the best validation macro-F1 in checkpoints/<name>/best.
"""
import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path

import matplotlib
import peft
import torch
import torch.nn.functional as F
import transformers
from peft import LoraConfig, get_peft_model

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from src.data import load_split, render_prompt, target_text
from src.metrics import full_report
from src.pipeline import load_model, predict_rows

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--name", help="defaults to the config file's name")
parser.add_argument("--limit", type=int, help="use only the first N rows, for a quick check")
parser.add_argument("--epochs", type=int, help="override the config")
args = parser.parse_args()
cfg = json.loads(Path(args.config).read_text())
cfg["epochs"] = args.epochs or cfg["epochs"]
name = args.name or Path(args.config).stem
transformers.set_seed(cfg["seed"])
order = random.Random(cfg["seed"])  # per-epoch shuffle order


def encode(tok, df) -> list[tuple[list[int], list[int]]]:
    """Tokenize examples as (tokens, labels), with -100 masking the prompt."""
    items = []
    for r in df.itertuples():
        prompt = tok(render_prompt(tok, r.note, r.criterion_type, r.criterion_text), add_special_tokens=False).input_ids
        answer_text = target_text(r.expert_eligibility, json.loads(r.expert_sentences)) + tok.eos_token
        answer = tok(answer_text, add_special_tokens=False).input_ids
        items.append((prompt + answer, [-100] * len(prompt) + answer))
    return items


def collate(batch, pad_id: int) -> dict:
    """Left-pad a batch so the answers end at the same position."""
    width = max(len(tokens) for tokens, _ in batch)
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for tokens, labels in batch:
        gap = width - len(tokens)
        out["input_ids"].append([pad_id] * gap + tokens)
        out["labels"].append([-100] * gap + labels)
        out["attention_mask"].append([0] * gap + [1] * len(tokens))
    return {k: torch.tensor(v, device="cuda") for k, v in out.items()}


def batches(items: list, size: int):
    return (items[i:i + size] for i in range(0, len(items), size))


def loss_of(model, batch):
    """Cross-entropy on the answer tokens only."""
    labels = batch.pop("labels")
    # Full-sequence logits cost ~1.5 GB per batch, so keep only the answer positions.
    keep = (labels != -100).sum(dim=1).max().item() + 1
    # LoRA weights stay fp32 so small updates aren't rounded away.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**batch, use_cache=False, logits_to_keep=keep).logits
    # shift by one: each position predicts the next token
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(), labels[:, 1 - keep:].reshape(-1), ignore_index=-100)


tok, model = load_model(cfg["model"])
# Trades speed for memory on the backward pass.
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                                         target_modules=cfg["target_modules"], task_type="CAUSAL_LM"))
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable LoRA numbers: {trainable:,} of {sum(p.numel() for p in model.parameters()):,}")
assert trainable > 0

train_df, val_df = load_split("train"), load_split("val")
if args.limit:
    train_df, val_df = train_df.head(args.limit), val_df.head(args.limit)
train_items, val_items = encode(tok, train_df), encode(tok, val_df)
print("longest example:", max(len(t) for t, _ in train_items), "tokens")
tokens, labels = train_items[0]
graded = [t for t, l in zip(tokens, labels) if l != -100]
print(f"example 0: {len(tokens)} tokens, {len(graded)} graded -> {tok.decode(graded)!r}")

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"])
out = Path("checkpoints") / name
log_dir = Path("results/train")
log_dir.mkdir(parents=True, exist_ok=True)
(log_dir / f"{name}_config.json").write_text(json.dumps({
    **cfg, "trainable_params": trainable, "python": platform.python_version(), "torch": torch.__version__,
    "transformers": transformers.__version__, "peft": peft.__version__, "gpu": torch.cuda.get_device_name(0)}, indent=2))

history, best = [], -1.0
torch.cuda.reset_peak_memory_stats()
for epoch in range(1, cfg["epochs"] + 1):
    start = time.perf_counter()
    model.train()
    order.shuffle(train_items)
    total = 0.0
    for chunk in batches(train_items, cfg["batch_size"]):
        loss = loss_of(model, collate(chunk, tok.pad_token_id))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        total += loss.item() * len(chunk)
    train_loss = total / len(train_items)

    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        val_total = sum(loss_of(model, collate(c, tok.pad_token_id)).item() * len(c) for c in batches(val_items, 8))
        report = full_report(predict_rows(tok, model, val_df))
    row = {"epoch": epoch, "train_loss": round(train_loss, 4), "val_loss": round(val_total / len(val_items), 4),
           "val_macro_f1": round(report["overall"]["labels"]["macro_f1"], 4),
           "val_accuracy": round(report["overall"]["labels"]["accuracy"], 4),
           "val_evidence_micro_f1": round(report["overall"]["evidence"]["micro"]["f1"], 4),
           "val_unusable": round(report["parse_failure_rate"], 4),
           "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
           "seconds": round(time.perf_counter() - start)}
    history.append(row)
    print(row, flush=True)
    if row["val_macro_f1"] > best:  # select on validation only
        best = row["val_macro_f1"]
        model.save_pretrained(out / "best")

with open(log_dir / f"{name}_log.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, history[0].keys())
    writer.writeheader()
    writer.writerows(history)

fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.5))
epochs = [h["epoch"] for h in history]
a.plot(epochs, [h["train_loss"] for h in history], label="train loss")
a.plot(epochs, [h["val_loss"] for h in history], label="val loss")
a.set_xlabel("epoch"); a.legend()
b.plot(epochs, [h["val_macro_f1"] for h in history], label="val macro-F1")
b.set_xlabel("epoch"); b.legend()
fig.tight_layout()
fig.savefig(log_dir / f"{name}_curves.png", dpi=150)
print(f"best val macro-F1: {best} -> saved in {out / 'best'}")

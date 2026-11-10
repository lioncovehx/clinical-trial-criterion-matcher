# Criterion-Level Clinical Trial Matching

> **Educational screening prototype, not a clinical enrollment decision.**

Given a patient note and one trial criterion, the system predicts an eligibility label and cites the supporting patient sentences. It fine-tunes Qwen3-1.7B on the TrialGPT Criterion Annotations dataset, and includes hand-built KV-cached decoding, tiled attention with online softmax, greedy speculative decoding, and a browser app.

- Demo video: https://youtu.be/hrAsIXDdcZQ
- Fine-tuned adapter: https://huggingface.co/lioncovehx/qwen3-1.7b-trialgpt-lora
- Draft adapter (speculative decoding): https://huggingface.co/lioncovehx/qwen3-0.6b-trialgpt-draft-lora

## Results on the held-out test split

| Qwen3-1.7B | Accuracy | Macro-F1 | Macro-F1 (all 6 labels) | Evidence micro-F1 |
|---|---:|---:|---:|---:|
| Untrained | 0.238 | 0.280 | 0.234 | 0.356 |
| Fine-tuned (LoRA) | **0.738** | **0.718** | **0.598** | **0.615** |

The test split has no "excluded" rows, so that label is left out of the first macro-F1 and scored as 0 in the second.

## Requirements

- Linux with an NVIDIA GPU of at least 8 GB (tested: RTX 5060 Laptop GPU, driver 595.104, CUDA 13.0)
- Python 3.12
- About 15 GB of free disk space

## 1. Install

```bash
git clone <this repository>
cd <repository-folder>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` points pip at PyTorch's CUDA 13 package index.

## 2. Start the app

```bash
source .venv/bin/activate
uvicorn src.app.server:app --port 8000
```

Open **http://localhost:8000** once the terminal shows `Application startup complete`. The first start downloads both base models and both adapters from Hugging Face, about 5.5 GB. No code, notebook or IDE is needed after this step.

In the app: paste a patient note, add criteria (paste, add one by one, or fetch by NCT ID), pick a decoding mode, and run. Each result shows the label, cited sentences and any missing information. The **Live benchmark** tab compares the decoding modes on the current input, and you can download a report.

## 3. Reproduce the experiments (optional)

Download the models and dataset into the project:

```bash
hf download Qwen/Qwen3-1.7B --local-dir models/Qwen3-1.7B
hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B
hf download HuggingFaceTB/SmolLM2-1.7B-Instruct --local-dir models/SmolLM2-1.7B-Instruct --exclude "onnx/*"
hf download ncbi/TrialGPT-Criterion-Annotations --repo-type dataset --local-dir data/raw
```

Run from the project folder:

| Step | Command |
|---|---|
| Explore the data | `python scripts/explore_data.py` |
| Build the split (seed 42, 37/8/8) | `python scripts/make_split.py` |
| Core tests | `python -m tests.test_core` and `python -m tests.test_app_text` |
| Baselines on validation | `python -m scripts.evaluate --model models/Qwen3-1.7B --name qwen_base` |
| Fine-tune (best config) | `python -m scripts.train_lora --config configs/lora_v2_lowlr.json` |
| Train the draft | `python -m scripts.train_lora --config configs/lora_draft.json` |
| Evaluate the adapter | `python -m scripts.evaluate --model models/Qwen3-1.7B --adapter checkpoints/lora_v2_lowlr/best --name qwen_lora_v2` |
| Final test run (once only) | add `--split test` to the two evaluate commands |
| Error analysis | `python -m scripts.error_analysis` |
| KV cache | `python -m tests.test_kv_cache`, then `python -m scripts.bench_kv_cache` |
| Tiled attention | `python -m tests.test_attention`, `python -m scripts.bench_attention`, `python -m scripts.tiled_in_model` |
| Speculative decoding | `python -m tests.test_speculative`, then `python -m scripts.bench_speculative` |

Benchmarks are sensitive to power mode, so keep a laptop plugged in and close other GPU programs.

## Project layout

```
configs/    training settings
data/       split_manifest.json (patient IDs, counts, hash); data/raw is the downloaded dataset
src/        data.py (split, prompt, parser), metrics.py, pipeline.py, decoding.py, attention.py, app/
scripts/    one entry point per experiment
tests/      checks with hand-worked answers and reference comparisons
results/    predictions, metrics, training logs, benchmark logs, figures, screenshots
```

Not in git: `models/`, `data/raw/`, `checkpoints/`, `.venv/`.

## Evaluation conventions

These are defined in `src/metrics.py`.

- Unusable model output counts as a wrong "invalid" prediction and is never dropped.
- Primary macro-F1 averages the labels present in the gold data; a label with no examples is shown as n/a.
- Evidence is reported as micro, macro over records with expert evidence, macro over all records (empty/empty = 1, empty gold with citations = 0), and empty agreement.
- Decoding is greedy, up to 64 new tokens, non-thinking mode, identical before and after fine-tuning.

## Limits

The dataset is small (53 patients), the test split has no "excluded" rows, and results come from one split and one training seed. The overall verdict in the app comes from a fixed rule shown on the page, and it hasn't been validated against patient outcomes. The model also struggles with clinical inference and negated rules. Don't use it for real clinical decisions.

## Credits

- Dataset: [TrialGPT Criterion Annotations](https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations)
- Models: [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), [SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)
- Built with help from Claude Code (Anthropic).

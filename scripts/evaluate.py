"""Score a model on a split. Saves every prediction and all the metrics to results/.

Untrained:  python -m scripts.evaluate --model models/Qwen3-1.7B --name qwen_base
Trained:    python -m scripts.evaluate --model models/Qwen3-1.7B --adapter <folder> --name qwen_lora
This is the only script allowed to open the test split (--split test), and only at the very end.
"""
import argparse
import json
from pathlib import Path

from src.data import load_split
from src.metrics import full_report
from src.pipeline import load_model, predict_rows

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--adapter")
parser.add_argument("--name", required=True)
parser.add_argument("--split", default="val", choices=["val", "test"])
parser.add_argument("--limit", type=int, help="only use the first N rows, for a quick trial")
args = parser.parse_args()

df = load_split(args.split, allow_test=args.split == "test")
if args.limit:
    df = df.head(args.limit)
tok, model = load_model(args.model, args.adapter)
records = predict_rows(tok, model, df)
report = full_report(records)

for folder, suffix, text in (("preds", "jsonl", "\n".join(json.dumps(r) for r in records)),
                             ("metrics", "json", json.dumps(report, indent=2))):
    path = Path("results") / folder / f"{args.name}_{args.split}.{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print("wrote", path)

overall = report["overall"]
print(f"\nrows: {len(records)}")
print(f"accuracy:             {overall['labels']['accuracy']:.3f}")
print(f"macro-F1:             {overall['labels']['macro_f1']:.3f}")
print(f"evidence micro-F1:    {overall['evidence']['micro']['f1']:.3f}")
print(f"unusable outputs:     {report['parse_failure_rate']:.1%}  {report['parse']}")
print(f"always-'{report['majority_baseline']['label']}' accuracy: {report['majority_baseline']['accuracy']:.3f}, "
      f"macro-F1: {report['majority_baseline']['macro_f1']:.3f}")

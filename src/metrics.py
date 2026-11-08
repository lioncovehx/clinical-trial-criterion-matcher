"""Label and evidence metrics. All conventions live here so the report and the app agree.

Label conventions
- A parse failure is the prediction "invalid", so it is always wrong and never dropped.
- Primary macro-F1 averages the labels that occur in the gold data of the group being scored.
  A label with no gold examples (e.g. "excluded" in test) has no defined recall, so it is
  listed as n/a. macro_f1_all_labels keeps those labels at 0, for comparison.

Evidence conventions (gold = expert sentence IDs, pred = cited IDs)
- micro: pools TP/FP/FN over all records; empty/empty adds nothing.
- macro_nonempty_gold: averages records with non-empty gold only. The primary macro number.
- macro_all: every record; empty/empty scores 1, empty gold with citations scores 0.
- empty_agreement: of the empty-gold records, the share where the model also cited nothing.
"""
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.data import ALL_LABELS, INVALID, LABELS


def label_metrics(gold: list[str], pred: list[str], label_space: list[str]) -> dict:
    supported = [l for l in label_space if l in gold]
    p, r, f, s = precision_recall_fscore_support(gold, pred, labels=label_space, average=None, zero_division=0)
    per_class = {}
    for i, label in enumerate(label_space):
        scores = dict(precision=float(p[i]), recall=float(r[i]), f1=float(f[i]), support=int(s[i]))
        per_class[label] = scores if s[i] else {"note": "n/a (no support)", "support": 0}
    matrix_labels = [*label_space, INVALID]
    return {
        "n": len(gold),
        "accuracy": float(accuracy_score(gold, pred)),
        "macro_f1": sum(per_class[l]["f1"] for l in supported) / len(supported) if supported else None,
        "macro_f1_all_labels": float(f.mean()),
        "per_class": per_class,
        "confusion_labels": matrix_labels,
        "confusion_matrix": confusion_matrix(gold, pred, labels=matrix_labels).tolist(),
    }


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def _mean(triples: list[tuple]) -> dict:
    n = len(triples)
    return {k: sum(t[i] for t in triples) / n if n else None for i, k in enumerate(("precision", "recall", "f1"))}


def evidence_metrics(gold: list[list[int]], pred: list[list[int]]) -> dict:
    tp = fp = fn = 0
    nonempty, every = [], []
    empty_total = empty_agree = 0
    for g, p in zip(map(set, gold), map(set, pred)):
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
        if g:
            nonempty.append(prf(len(g & p), len(p - g), len(g - p)))
            every.append(nonempty[-1])
        else:
            empty_total += 1
            empty_agree += not p
            every.append((1.0, 1.0, 1.0) if not p else (0.0, 0.0, 0.0))
    micro = prf(tp, fp, fn)
    return {
        "micro": dict(zip(("precision", "recall", "f1"), micro)),
        "macro_nonempty_gold": _mean(nonempty),
        "macro_all": _mean(every),
        "empty_agreement": empty_agree / empty_total if empty_total else None,
        "n_records": len(gold),
        "n_empty_gold": empty_total,
    }


def full_report(records: list[dict]) -> dict:
    """records need: criterion_type, gold_label, pred_label, gold_ids, pred_ids, status, hallucinated."""
    def group(rs, space):
        return {"labels": label_metrics([r["gold_label"] for r in rs], [r["pred_label"] for r in rs], space),
                "evidence": evidence_metrics([r["gold_ids"] for r in rs], [r["pred_ids"] for r in rs])}

    statuses = [r["status"] for r in records]
    majority = max(set(r["gold_label"] for r in records), key=[r["gold_label"] for r in records].count)
    return {
        "overall": group(records, ALL_LABELS),
        "inclusion": group([r for r in records if r["criterion_type"] == "inclusion"], LABELS["inclusion"]),
        "exclusion": group([r for r in records if r["criterion_type"] == "exclusion"], LABELS["exclusion"]),
        "parse": {s: statuses.count(s) for s in sorted(set(statuses))},
        "parse_failure_rate": sum(s != "ok" for s in statuses) / len(statuses),
        "hallucinated_id_count": sum(len(r["hallucinated"]) for r in records),
        "majority_baseline": {"label": majority,
                              **label_metrics([r["gold_label"] for r in records], [majority] * len(records), ALL_LABELS)},
    }

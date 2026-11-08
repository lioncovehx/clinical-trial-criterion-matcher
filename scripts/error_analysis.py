"""Categorize the final model's test errors and write up selected cases.

Run: python -m scripts.error_analysis
Writes results/error_analysis/summary.json and cases.md.
"""
import collections
import json
import re
from pathlib import Path

from src.data import load_split, sentences

# (annotation id, error kind, likely source, diagnosis)
CASES = [
    (405, "Missing information treated as a negative", "Model + labelling convention",
     "The note never mentions surgery. The expert says this can't be decided; the model takes the silence to mean the patient had none."),
    (425, "Labelling convention for unmentioned rules", "Dataset",
     "An APACHE II score can't be worked out from this note, and the note doesn't state one, so the model's caution is reasonable. The expert marks an unmentioned exclusion as not excluded here, but as not enough information elsewhere."),
    (766, "Over-cautious despite direct evidence", "Model",
     "The note says fever outright and the model cites the right sentence, yet still declines to decide."),
    (186, "Medical term needs interpreting", "Model",
     "Tachycardic means a heart rate above 100, so at least 90 is met. The model did not make that link and cited the blood-pressure sentence."),
    (188, "Numeric threshold needs reasoning", "Model",
     "A blood pressure of 80/60 shows systolic pressure is not being held at 90 or above. The model cites the right sentence without drawing the conclusion."),
    (834, "Not applicable vs not enough information", "Model",
     "A pregnancy test can't apply to a 4-year-old boy. The model treats this as missing data, not as a rule that doesn't apply."),
    (322, "Applicability, labelled inconsistently", "Dataset + model",
     "The patient is male. The expert says not excluded, citing his sex; the model says not applicable. Case 834 shows experts also use not applicable for this kind of rule."),
    (414, "Negated exclusion rule", "Model",
     "The rule excludes patients without lupus, and the note strongly suggests lupus, so the patient is not excluded. The double negative defeats the model, which cites lupus findings but won't decide."),
    (827, "Contradictory information in the note", "Model",
     "The boy is wheezing now but has no history of wheezing, and it began suddenly while playing, so asthma is neither confirmed nor ruled out. The model picks one side, citing the no-history sentence."),
    (410, "Unsupported evidence", "Model",
     "The cited sentences describe lupus symptoms, while the rule asks about other diseases not related to lupus. The citations don't support the answer."),
    (323, "Only part of a compound rule is met", "Model",
     "The rule needs immune deficiency and bone marrow failure in childhood. The note shows only immune deficiency, in a 48-year-old, but the model accepts it."),
    (413, "Right label, incomplete evidence", "Model",
     "The label is correct, but the model misses sentence 3, the kidney finding, which the expert also counts."),
    (384, "Right label, cites beyond the expert", "Labelling convention",
     "Not enough information is correct. The model cites the lab sentence, which is relevant but doesn't give the needed values; the expert cites nothing."),
]

records = [json.loads(line) for line in open("results/preds/qwen_lora_v2_test.jsonl")]
rows = load_split("test", allow_test=True).set_index("annotation_id")
by_id = {r["annotation_id"]: r for r in records}
wrong = [r for r in records if r["gold_label"] != r["pred_label"]]
has_number = lambda r: bool(re.search(r"\d", r["criterion"] or ""))
right = [r for r in records if r["gold_label"] == r["pred_label"]]
evidence_off = [r for r in right if set(r["gold_ids"]) != set(r["pred_ids"])]
boundary = {"not excluded", "not enough information"}
on_boundary = [r for r in wrong if {r["gold_label"], r["pred_label"]} == boundary]
trial_of = lambda r: rows.loc[r["annotation_id"], "trial_id"]
unmentioned = rows[(rows.criterion_type == "exclusion") & (rows.expert_sentences == "[]") & rows.expert_eligibility.isin(boundary)]

summary = {
    "total": len(records),
    "label_errors": len(wrong),
    "error_pairs": [{"gold": g, "predicted": p, "count": n} for (g, p), n in collections.Counter((r["gold_label"], r["pred_label"]) for r in wrong).most_common()],
    "not_excluded_vs_not_enough_info_errors": len(on_boundary),
    "errors_by_trial_top5": [{"trial": t, "errors": n} for t, n in collections.Counter(map(trial_of, wrong)).most_common(5)],
    "unmentioned_exclusion_labels": unmentioned.expert_eligibility.value_counts().to_dict(),
    "unmentioned_exclusion_by_trial": {t: g.expert_eligibility.value_counts().to_dict() for t, g in unmentioned.groupby("trial_id")},
    "errors_by_type": {t: sum(r["criterion_type"] == t for r in wrong) for t in ("inclusion", "exclusion")},
    "rows_by_type": {t: sum(r["criterion_type"] == t for r in records) for t in ("inclusion", "exclusion")},
    "numeric_criteria": {"rows": sum(map(has_number, records)), "errors": sum(map(has_number, wrong))},
    "non_numeric_criteria": {"rows": sum(not has_number(r) for r in records), "errors": sum(not has_number(r) for r in wrong)},
    "right_label_evidence_differs": len(evidence_off),
    "right_label_cited_when_expert_cited_none": sum(bool(r["pred_ids"]) and not r["gold_ids"] for r in evidence_off),
    "right_label_cited_none_when_expert_cited": sum(bool(r["gold_ids"]) and not r["pred_ids"] for r in evidence_off),
    "right_label_partial_overlap": sum(bool(r["gold_ids"]) and bool(r["pred_ids"]) for r in evidence_off),
    "parse_failures": sum(r["status"] != "ok" for r in records),
    "hallucinated_ids": sum(len(r["hallucinated"]) for r in records),
    "hand_analyzed_cases": len(CASES),
    "distinct_error_kinds": len({c[1] for c in CASES}),
}
out = Path("results/error_analysis")
out.mkdir(parents=True, exist_ok=True)
(out / "summary.json").write_text(json.dumps(summary, indent=2))

lines = ["# Error analysis: final model on the test split", ""]
for i, (case_id, kind, owner, why) in enumerate(CASES, 1):
    r, sents = by_id[case_id], sentences(rows.loc[case_id, "note"])
    cite = lambda ids: "none" if not ids else "; ".join(f'[{s}] "{sents[s]}"' for s in ids)
    lines += [f"## {i}. {kind}", "", f"- Annotation: {case_id} ({r['criterion_type']})", f"- Criterion: {r['criterion']}",
              f"- Expert: **{r['gold_label']}**, citing {cite(r['gold_ids'])}", f"- Model: **{r['pred_label']}**, citing {cite(r['pred_ids'])}",
              f"- Source of the problem: {owner}", f"- Why: {why}", ""]
(out / "cases.md").write_text("\n".join(lines))
print(json.dumps({k: v for k, v in summary.items() if not k.startswith("unmentioned_exclusion_by")}, indent=1))

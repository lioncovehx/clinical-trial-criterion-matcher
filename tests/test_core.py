"""Hand-computed checks for the parser and the metrics. Run: python -m tests.test_core"""
from math import isclose

from src.data import INVALID, LABELS, parse_output, sentences
from src.metrics import evidence_metrics, label_metrics

# --- Parser ---
ids = {0, 1, 2}
ok = parse_output('{"evidence": [2, 0], "label": "included"}', "inclusion", ids)
assert ok == {"status": "ok", "label": "included", "ids": [0, 2], "hallucinated": []}
assert parse_output('```json\n{"evidence": [], "label": "Not Excluded"}\n```', "exclusion", ids)["label"] == "not excluded"
assert parse_output("no json here", "inclusion", ids)["status"] == "invalid_json"
assert parse_output('{"evidence": 3, "label": "included"}', "inclusion", ids)["status"] == "invalid_schema"
wrong_type = parse_output('{"evidence": [1], "label": "excluded"}', "inclusion", ids)
assert wrong_type["status"] == "invalid_label" and wrong_type["label"] == INVALID
fake = parse_output('{"evidence": [1, 9, "x"], "label": "included"}', "inclusion", ids)
assert fake["ids"] == [1] and fake["hallucinated"] == [9, "x"]
assert sentences("0. Hello there.\n1. Age is 3.5 years.") == {0: "Hello there.", 1: "Age is 3.5 years."}

# --- Labels. Gold has 2 included, 1 not included; "not applicable" and "NEI" have no gold.
# included: TP=1 FP=0 FN=1 -> P=1, R=.5, F1=2/3.  not included: TP=1 FP=1 FN=0 -> P=.5, R=1, F1=2/3.
space = LABELS["inclusion"]
m = label_metrics(["included", "included", "not included"], ["included", "not included", "not included"], space)
assert isclose(m["accuracy"], 2 / 3)
assert isclose(m["per_class"]["included"]["f1"], 2 / 3) and isclose(m["per_class"]["not included"]["precision"], 0.5)
assert isclose(m["macro_f1"], 2 / 3)  # averaged over the 2 labels that have support
assert isclose(m["macro_f1_all_labels"], (2 / 3 + 2 / 3) / 4)  # the 2 unsupported labels count as 0
assert m["per_class"]["not applicable"]["support"] == 0 and "f1" not in m["per_class"]["not applicable"]

# An invalid prediction is wrong and shows up in the confusion matrix's last column.
m = label_metrics(["included", "included"], ["included", INVALID], space)
assert m["accuracy"] == 0.5 and m["confusion_matrix"][0][-1] == 1 and isclose(m["per_class"]["included"]["recall"], 0.5)

# --- Evidence ---
# r1 g={0,1} p={1,2}: TP1 FP1 FN1, P=R=F=.5 | r2 empty/empty | r3 empty/{4}: FP1 | r4 g={3} p={}: FN1, all 0
gold, pred = [[0, 1], [], [], [3]], [[1, 2], [], [4], []]
e = evidence_metrics(gold, pred)
# Micro TP=1 FP=2 FN=2 -> P=R=F=1/3
assert all(isclose(e["micro"][k], 1 / 3) for k in ("precision", "recall", "f1"))
# Non-empty gold is r1 and r4: F1 = (.5 + 0) / 2
assert isclose(e["macro_nonempty_gold"]["f1"], 0.25)
# All records: F1 = (.5 + 1 + 0 + 0) / 4
assert isclose(e["macro_all"]["f1"], 0.375)
assert e["empty_agreement"] == 0.5 and e["n_empty_gold"] == 2

print("all core tests passed")

"""Loading the fixed splits, building prompts, and parsing model output.

Every part of the system (training, evaluation, baseline, app) must call these same
functions, so the prompt text can never differ between them.
"""
import json
import re
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/raw/data/train-00000-of-00001.parquet")
MANIFEST_PATH = Path("data/split_manifest.json")

COMMON = ["not applicable", "not enough information"]
LABELS = {"inclusion": ["included", "not included", *COMMON],
          "exclusion": ["excluded", "not excluded", *COMMON]}
ALL_LABELS = ["included", "not included", "excluded", "not excluded", *COMMON]
INVALID = "invalid"  # stands in for an unusable output, so it is scored as wrong and never dropped

SYSTEM = """You screen patients for clinical trials. You get a patient note with numbered sentences and one trial criterion.
Choose one label and list the IDs of the sentences that support it. Cite no IDs if no sentence is relevant.

Labels for an inclusion criterion:
- included: the note shows the patient meets the criterion
- not included: the note shows the patient does not meet the criterion
Labels for an exclusion criterion:
- excluded: the note shows the patient meets the criterion, so the patient is excluded
- not excluded: the patient does not meet the criterion
Labels for both:
- not enough information: the note does not say enough to decide
- not applicable: the criterion does not apply to this patient

Reply with JSON only: {"evidence": [sentence IDs], "label": "<label>"}"""


def load_split(split: str, allow_test: bool = False) -> pd.DataFrame:
    """Rows for one split. The test split needs allow_test=True, set only by the final evaluation."""
    if split == "test" and not allow_test:
        raise PermissionError("the test split is reserved for the final evaluation")
    patients = json.loads(MANIFEST_PATH.read_text())[split]["patient_ids"]
    df = pd.read_parquet(DATA_PATH)
    return df[df.patient_id.isin(patients)].reset_index(drop=True)


def sentences(note: str) -> dict[int, str]:
    """{0: "first sentence", 1: ...} from a "0. text\\n1. text" note."""
    matches = (re.match(r"(\d+)\.\s*(.*)", line) for line in note.split("\n"))
    return {int(m[1]): m[2] for m in matches if m}


def clean_criterion(text) -> str:
    # Annotation 883 has a null criterion: its trial lists no exclusion criteria.
    return text if isinstance(text, str) and text.strip() else "(no criterion text was provided)"


def user_message(note: str, criterion_type: str, criterion_text) -> str:
    return (f"Patient note:\n{note}\n\nCriterion type: {criterion_type}\n"
            f"Criterion: {clean_criterion(criterion_text)}\n"
            f"Allowed labels: {', '.join(LABELS[criterion_type])}")


def render_prompt(tok, note: str, criterion_type: str, criterion_text) -> str:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_message(note, criterion_type, criterion_text)}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def target_text(label: str, ids: list[int]) -> str:
    """The answer the model is trained to give. Evidence comes first, so the label is conditioned on it."""
    return json.dumps({"evidence": sorted(ids), "label": label})


def parse_output(text: str, criterion_type: str, valid_ids) -> dict:
    """Turn raw model text into {status, label, ids, hallucinated}. Never raises."""
    bad = {"label": INVALID, "ids": [], "hallucinated": []}
    match = re.search(r"\{.*?\}", text, re.DOTALL)  # tolerate text or code fences around the object
    if not match:
        return {"status": "invalid_json", **bad}
    try:
        obj = json.loads(match[0])
    except json.JSONDecodeError:
        return {"status": "invalid_json", **bad}
    if not isinstance(obj, dict) or not isinstance(obj.get("evidence"), list):
        return {"status": "invalid_schema", **bad}
    cited = [i for i in obj["evidence"] if isinstance(i, int) and not isinstance(i, bool)]
    ids = sorted({i for i in cited if i in valid_ids})
    # Non-integer entries count as hallucinated too: they can't point at any sentence.
    hallucinated = [i for i in obj["evidence"] if i not in ids]
    label = obj.get("label")
    label = label.strip().lower() if isinstance(label, str) else None
    if label not in LABELS[criterion_type]:
        return {"status": "invalid_label", "label": INVALID, "ids": ids, "hallucinated": hallucinated}
    return {"status": "ok", "label": label, "ids": ids, "hallucinated": hallucinated}

"""Print summary statistics for the dataset.

Run from the repo root:  python scripts/explore_data.py
"""
import json
import re
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

DATA_PATH = Path("data/raw/data/train-00000-of-00001.parquet")
MODEL_DIR = "models/Qwen3-1.7B"


def header(title: str) -> None:
    print(f"\n===== {title} =====")


def parse_ids(s: str) -> list[int]:
    # stored as a JSON string, e.g. "[0, 1, 2]"
    return json.loads(s)


def sentences(note: str) -> dict[int, str]:
    # "0. text\n1. text" -> {0: "text", 1: "text"}
    matches = (re.match(r"(\d+)\.\s*(.*)", line) for line in note.split("\n"))
    return {int(m[1]): m[2] for m in matches if m}


def basic_stats(df: pd.DataFrame) -> None:
    header("Basic stats")
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print("unique patients:", df.patient_id.nunique())
    print("unique trials:", df.trial_id.nunique())
    # every patient should have exactly one note
    print("max distinct notes for one patient:", df.groupby("patient_id").note.nunique().max())


def label_table(df: pd.DataFrame) -> None:
    header("Labels by criterion type")
    print(pd.crosstab(df.expert_eligibility, df.criterion_type, margins=True))


def evidence_stats(df: pd.DataFrame) -> None:
    header("Evidence")
    n_cited = df.expert_sentences.map(parse_ids).map(len)
    print("rows with empty evidence:", (n_cited == 0).sum())
    empty = n_cited == 0
    print(pd.DataFrame({"empty": empty.groupby(df.expert_eligibility).sum(),
                        "total": empty.groupby(df.expert_eligibility).size()}))
    print("most sentences cited in one row:", n_cited.max())


def note_stats(df: pd.DataFrame) -> None:
    header("Note and criterion lengths")
    # one row per patient
    notes = df.drop_duplicates("patient_id").note
    n_sent = notes.map(lambda n: len(sentences(n)))
    print("sentences per note: min", n_sent.min(), "median", n_sent.median(), "max", n_sent.max())
    print("note characters: min", notes.str.len().min(), "max", notes.str.len().max())
    print("criterion characters: median", df.criterion_text.str.len().median(),
          "max", df.criterion_text.str.len().max())
    print("last sentence of 3 notes (truncation must never remove these):")
    for note in notes.head(3):
        print("  -", note.split("\n")[-1])


def show_examples(df: pd.DataFrame, n: int = 3) -> None:
    header(f"{n} examples that cite evidence")
    cited = df[df.expert_sentences.map(parse_ids).map(len) > 0]
    for _, row in cited.head(n).iterrows():
        print(f"\n[{row.annotation_id}] {row.criterion_type}: {row.criterion_text}")
        print("label:", row.expert_eligibility, "| evidence IDs:", row.expert_sentences)
        sents = sentences(row.note)
        for i in parse_ids(row.expert_sentences):
            print(f"  {i}: {sents[i]}")


def token_lengths(df: pd.DataFrame) -> None:
    header("Prompt token lengths (simple prompt)")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    lengths = []
    for _, row in df.iterrows():
        user = f"Patient note:\n{row.note}\n\nCriterion type: {row.criterion_type}\nCriterion: {row.criterion_text}"
        text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        lengths.append(len(tok(text).input_ids))
    s = pd.Series(lengths)
    print("min", s.min(), "median", s.median(), "max", s.max())


def criterion_repeats(df: pd.DataFrame) -> None:
    header("Criterion repetition")
    counts = df.criterion_text.value_counts()
    print("unique criteria:", len(counts), "of", len(df), "rows")
    print(counts.head(5).to_string())


def look_at_not_applicable(df: pd.DataFrame, n: int = 10) -> None:
    header(f"{n} 'not applicable' criteria")
    for _, row in df[df.expert_eligibility == "not applicable"].head(n).iterrows():
        print(f"- ({row.criterion_type}) {row.criterion_text[:150]}")


def look_at_gpt4_incorrect(df: pd.DataFrame) -> None:
    # inspection only; gpt4_* columns never reach a prompt
    bad = df[df.explanation_correctness == "Incorrect"]
    header(f"GPT-4 'Incorrect' explanations: {len(bad)} rows")
    print(pd.crosstab(bad.expert_eligibility, bad.gpt4_eligibility))


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    basic_stats(df)
    label_table(df)
    evidence_stats(df)
    note_stats(df)
    show_examples(df)
    token_lengths(df)
    criterion_repeats(df)
    look_at_not_applicable(df)
    look_at_gpt4_incorrect(df)


if __name__ == "__main__":
    main()

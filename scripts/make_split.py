"""Patient-level train/val/test split and manifest.

Sorted patient IDs, shuffled with numpy.random.default_rng(42): 37 train, 8 val, 8 test.

Run from the repo root:  python scripts/make_split.py
Writes data/split_manifest.json.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path("data/raw/data/train-00000-of-00001.parquet")
MANIFEST_PATH = Path("data/split_manifest.json")
SEED = 42
N_TRAIN, N_VAL, N_TEST = 37, 8, 8

# Expected row counts for this dataset and seed.
EXPECTED_ROWS = {"train": 603, "val": 206, "test": 206}


def sha256_file(path: Path) -> str:
    # identifies the exact dataset file
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sorted_patient_ids(df: pd.DataFrame) -> list[str]:
    # sort first so the shuffle doesn't depend on row order
    ids = sorted(str(p) for p in df.patient_id.unique())
    assert len(ids) == N_TRAIN + N_VAL + N_TEST, f"expected 53 patients, got {len(ids)}"
    return ids


def check_shuffle_methods(ids: list[str]) -> None:
    # fresh generator per method, same starting state
    a = np.array(ids)
    np.random.default_rng(SEED).shuffle(a)
    b = np.random.default_rng(SEED).permutation(np.array(ids))
    c = list(ids)
    np.random.default_rng(SEED).shuffle(c)
    assert a.tolist() == b.tolist(), "shuffle and permutation disagree on a numpy array"
    print("shuffle(array) == permutation(array): True")
    print("shuffle(array) == shuffle(list):", a.tolist() == c)


def split_patients(ids: list[str]) -> dict[str, list[str]]:
    rng = np.random.default_rng(SEED)
    shuffled = np.array(ids)
    rng.shuffle(shuffled)  # in place
    shuffled = shuffled.tolist()  # numpy strings don't serialize to JSON
    return {
        "train": shuffled[:N_TRAIN],
        "val": shuffled[N_TRAIN:N_TRAIN + N_VAL],
        "test": shuffled[N_TRAIN + N_VAL:],
    }


def rows_for(df: pd.DataFrame, patients: list[str]) -> pd.DataFrame:
    return df[df.patient_id.isin(patients)]


def label_distribution(df: pd.DataFrame, patients: list[str]) -> dict[str, int]:
    counts = rows_for(df, patients).expert_eligibility.value_counts()
    return {label: int(n) for label, n in counts.items()}


def leakage_checks(df: pd.DataFrame, splits: dict[str, list[str]]) -> None:
    train, val, test = (set(splits[k]) for k in ("train", "val", "test"))
    assert not (train & val or train & test or val & test), "a patient appears in two splits"
    assert train | val | test == set(df.patient_id), "the splits don't cover every patient"
    assert sum(len(rows_for(df, p)) for p in splits.values()) == len(df), "row counts don't add up"

    # GPT-4 columns are never read by the prompt builder. A substring check here would misfire:
    # annotation 604's GPT-4 explanation just quotes the note.
    # Test-split access is guarded in src.data.load_split.
    print("leakage checks passed")


def build_manifest(df: pd.DataFrame, splits: dict[str, list[str]]) -> dict:
    manifest = {
        "source_file": str(DATA_PATH),
        "source_sha256": sha256_file(DATA_PATH),
        "seed": SEED,
        "procedure": "sorted(unique patient_id as str) -> numpy default_rng(42).shuffle(array) -> first 37 train, next 8 val, last 8 test",
        "versions": {"numpy": np.__version__, "pandas": pd.__version__},
    }
    for name, patients in splits.items():
        rows = rows_for(df, patients)
        manifest[name] = {
            "n_patients": len(patients),
            "n_rows": len(rows),
            "label_distribution": label_distribution(df, patients),
            "patient_ids": patients,
            "annotation_ids": [int(a) if isinstance(a, np.integer) else a for a in rows.annotation_id],
        }
    return manifest


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    ids = sorted_patient_ids(df)
    check_shuffle_methods(ids)
    splits = split_patients(ids)
    leakage_checks(df, splits)
    manifest = build_manifest(df, splits)

    # Sanity-check the split against the known counts.
    for name, expected in EXPECTED_ROWS.items():
        got = manifest[name]["n_rows"]
        assert got == expected, f"{name}: {got} rows, expected {expected}"
    assert manifest["test"]["label_distribution"].get("excluded", 0) == 0

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {MANIFEST_PATH}")
    for name in ("train", "val", "test"):
        print(name, manifest[name]["n_patients"], "patients,", manifest[name]["n_rows"], "rows,",
              manifest[name]["label_distribution"])


if __name__ == "__main__":
    main()

"""Turning free text from the browser into numbered sentences and criteria."""
import re

ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "prof", "e.g", "i.e", "vs", "fig", "approx", "st", "pt", "pts", "hx", "dx", "tx"}
HEADER = re.compile(r"(inclusion|exclusion)\s+criteria", re.IGNORECASE)
BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
EMPTY_ITEMS = {"none", "n/a", "na", "not applicable"}


def split_sentences(text: str) -> list[str]:
    """Split on sentence ends, keeping decimals ("3.5 mg") and abbreviations ("Dr. Lee") together."""
    result = []
    for line in text.splitlines():
        line = re.sub(r"^\s*\d+[.)]\s+", "", line).strip()  # a note that is already numbered
        if not line:
            continue
        current = ""
        # A break needs whitespace after the punctuation and a capital or digit next.
        for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", line):
            current = f"{current} {part}" if current else part
            if current.endswith(".") and current.rsplit(None, 1)[-1].rstrip(".").lower() in ABBREVIATIONS:
                continue
            result.append(current)
            current = ""
        if current:
            result.append(current)
    return result


def number_note(sentences: list[str]) -> str:
    """The same "0. text" format the model was trained on."""
    return "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))


def parse_criteria(text: str) -> list[dict]:
    """Read pasted text with "Inclusion Criteria:" / "Exclusion Criteria:" headers, one item per line."""
    items, current = [], None
    for line in text.splitlines():
        if not line.strip():
            continue
        header = HEADER.search(line)
        if header and len(line.strip()) < 60:
            current = header[1].lower()
            continue
        # ClinicalTrials.gov escapes symbols as Markdown, e.g. "\\>10" for ">10".
        item = re.sub(r"\\([<>=*_~^-])", r"\1", BULLET.sub("", line)).strip()
        if current and item and item.lower() not in EMPTY_ITEMS:
            items.append({"type": current, "text": item})
    if not items:
        raise ValueError('No criteria found. Use lines under "Inclusion Criteria:" and "Exclusion Criteria:" headers.')
    return items

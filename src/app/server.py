"""Browser app for per-criterion screening, decoder comparison, and report export.

Start from the repo root:  uvicorn src.app.server:app --port 8000
Educational screening prototype, not a clinical enrollment decision.
"""
import hashlib
import html
import json
import re
import statistics
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, field_validator

from src.app.text import number_note, parse_criteria, split_sentences
from src.attention import tiled_attention_in
from src.data import render_prompt
from src.decoding import kv_cache_greedy, no_cache_greedy, speculative_greedy
from src.pipeline import load_model, make_record

DISCLAIMER = "Educational screening prototype, not a clinical enrollment decision."
RULE = ("Preliminary rule: any 'excluded' or 'not included' -> Likely ineligible. Otherwise, any 'not enough information' "
        "or unreadable answer -> Needs review. Otherwise, if at least one criterion applies -> Potentially eligible. "
        "'not applicable' is neutral. If nothing applies -> Needs review. Absence of evidence is never treated as eligible.")
MAX_NEW_TOKENS, MAX_PROMPT_TOKENS = 64, 4000
Mode = Literal["no_cache", "kv_cache", "speculative", "tiled"]


def resolve(local: str, hub: str) -> str:
    """Prefer the local copy, otherwise use the Hugging Face Hub."""
    return local if Path(local).exists() else hub


TARGET, TARGET_ADAPTER = resolve("models/Qwen3-1.7B", "Qwen/Qwen3-1.7B"), resolve("checkpoints/lora_v2_lowlr/best", "lioncovehx/qwen3-1.7b-trialgpt-lora")
DRAFT, DRAFT_ADAPTER = resolve("models/Qwen3-0.6B", "Qwen/Qwen3-0.6B"), resolve("checkpoints/lora_draft/best", "lioncovehx/qwen3-0.6b-trialgpt-draft-lora")

state = {"tok": None, "target": None, "draft": None, "error": None}
jobs: dict[str, dict] = {}  # ponytail: kept in memory and lost on restart; use a database for real multi-user history
# One worker serializes GPU jobs, so concurrent requests queue.
gpu = ThreadPoolExecutor(max_workers=1)


@asynccontextmanager
async def lifespan(_app):
    try:  # load once at startup
        state["tok"], state["target"] = load_model(TARGET, TARGET_ADAPTER)
        _, state["draft"] = load_model(DRAFT, DRAFT_ADAPTER)
    except Exception as e:  # keep serving so the page can show the error
        state["error"] = f"{type(e).__name__}: {e}"
    yield
    gpu.shutdown(wait=False)


app = FastAPI(title="Clinical Trial Criterion Screening", lifespan=lifespan)


class Criterion(BaseModel):
    type: Literal["inclusion", "exclusion"]
    text: str = Field(min_length=1, max_length=4000)

    @field_validator("text")
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError("A criterion can't be blank.")
        return value.strip()


class NoteRequest(BaseModel):
    note: str = Field(max_length=30000)

    @field_validator("note")
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError("Please enter a patient note.")
        return value


class TextRequest(BaseModel):
    text: str = Field(max_length=100000)


class ScreenRequest(NoteRequest):
    criteria: list[Criterion] = Field(min_length=1, max_length=60)
    mode: Mode = "kv_cache"
    k: int = Field(default=6, ge=1, le=8)


class BenchmarkRequest(NoteRequest):
    criterion: Criterion
    modes: list[Mode] = Field(min_length=1)
    k: int = Field(default=6, ge=1, le=8)


def require_model() -> None:
    if state["error"]:
        raise HTTPException(503, f"The model failed to load: {state['error']}")
    if state["target"] is None:
        raise HTTPException(503, "The model is still loading. Please try again in a moment.")


def prepare_note(note: str) -> tuple[list[str], str]:
    parts = split_sentences(note)
    if not parts:
        raise HTTPException(422, "No sentences were found in the patient note.")
    return parts, number_note(parts)


def build_ids(numbered: str, criterion: dict):
    prompt = render_prompt(state["tok"], numbered, criterion["type"], criterion["text"])
    ids = state["tok"](prompt, return_tensors="pt").input_ids.to("cuda")
    if ids.shape[1] > MAX_PROMPT_TOKENS:
        raise ValueError(f"This input needs {ids.shape[1]} tokens; the limit is {MAX_PROMPT_TOKENS}. Please shorten the note.")
    return ids


def decode(ids, mode: str, k: int) -> dict:
    """Run one decoder and return its tokens, time, and acceptance rate."""
    target, draft, eos = state["target"], state["draft"], {state["tok"].eos_token_id}
    if mode == "speculative":
        out = speculative_greedy(target, draft, ids, MAX_NEW_TOKENS, k, eos)
        return {"tokens": out["tokens"], "seconds": out["total_s"], "accept_rate": out["accepted"] / max(out["drafted"], 1)}
    if mode == "no_cache":
        out = no_cache_greedy(target, ids, MAX_NEW_TOKENS, eos)
    elif mode == "tiled":
        with tiled_attention_in(target):
            out = kv_cache_greedy(target, ids, MAX_NEW_TOKENS, eos)
    else:
        out = kv_cache_greedy(target, ids, MAX_NEW_TOKENS, eos)
    return {"tokens": out["tokens"], "seconds": out["prefill_s"] + out["decode_s"], "accept_rate": None}


def aggregate(results: list[dict]) -> str:
    labels = [r["pred_label"] for r in results]
    if any(l in ("excluded", "not included") for l in labels):
        return "Likely ineligible"
    if any(l in ("not enough information", "invalid") for l in labels):
        return "Needs review"
    return "Potentially eligible" if any(l in ("included", "not excluded") for l in labels) else "Needs review"


def fail(job: dict, message: str) -> None:
    job["status"], job["error"] = "failed", message
    torch.cuda.empty_cache()


def run_screen(job: dict) -> None:
    job["status"] = "running"
    try:
        for criterion in job["criteria"]:
            ids = build_ids(job["numbered"], criterion)
            out = decode(ids, job["mode"], job["k"])
            raw = state["tok"].decode(out["tokens"], skip_special_tokens=True)
            record = make_record(raw, job["numbered"], criterion["type"], criterion["text"])
            job["results"].append({**record, "seconds": round(out["seconds"], 3), "tokens": len(out["tokens"]),
                                   "tokens_per_s": round(len(out["tokens"]) / out["seconds"], 1), "accept_rate": out["accept_rate"]})
        job["aggregate"], job["status"] = aggregate(job["results"]), "done"
    except torch.OutOfMemoryError:
        fail(job, "The GPU ran out of memory. Close other GPU programs or shorten the note, then try again.")
    except ValueError as e:
        fail(job, str(e))
    except Exception as e:  # keep the page usable on any failure
        fail(job, f"Unexpected error: {type(e).__name__}: {e}")


def run_benchmark(job: dict) -> None:
    job["status"] = "running"
    try:
        ids = build_ids(job["numbered"], job["criterion"])
        # KV-cache tokens are the reference for the equality check.
        reference = decode(ids, "kv_cache", job["k"])["tokens"]
        for mode in job["modes"]:
            torch.cuda.reset_peak_memory_stats()
            runs = [decode(ids, mode, job["k"]) for _ in range(3)]
            seconds = statistics.median(r["seconds"] for r in runs)
            job["rows"].append({"mode": mode, "latency_ms": round(seconds * 1000, 1), "tokens": len(runs[0]["tokens"]),
                                "tokens_per_s": round(len(runs[0]["tokens"]) / seconds, 1),
                                "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                                "accept_rate": None if runs[0]["accept_rate"] is None else round(runs[0]["accept_rate"], 3),
                                "identical_to_reference": runs[0]["tokens"] == reference})
        job["status"] = "done"
    except torch.OutOfMemoryError:
        fail(job, "The GPU ran out of memory. Close other GPU programs or shorten the note, then try again.")
    except ValueError as e:
        fail(job, str(e))
    except Exception as e:
        fail(job, f"Unexpected error: {type(e).__name__}: {e}")


def start(job: dict, worker) -> dict:
    job.update(id=uuid.uuid4().hex, status="queued", error=None, created=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    jobs[job["id"]] = job
    gpu.submit(worker, job)
    return {"job_id": job["id"]}


@app.get("/")
def page():
    # no-cache so browsers revalidate the page
    return FileResponse(Path(__file__).parent / "static" / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/health")
def health():
    return {"ready": state["target"] is not None and state["error"] is None, "error": state["error"], "disclaimer": DISCLAIMER, "rule": RULE,
            "target": TARGET, "target_adapter": TARGET_ADAPTER, "draft": DRAFT, "draft_adapter": DRAFT_ADAPTER,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


@app.post("/split")
def split(request: NoteRequest):
    return {"sentences": prepare_note(request.note)[0]}


@app.post("/criteria/parse")
def criteria_parse(request: TextRequest):
    try:
        return {"criteria": parse_criteria(request.text)}
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/trial/{nct_id}")
def trial(nct_id: str):
    if not re.fullmatch(r"NCT\d{8}", nct_id):
        raise HTTPException(422, "A trial ID looks like NCT01234567: the letters NCT and 8 digits.")
    url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}?fields=EligibilityCriteria|BriefTitle"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=20) as reply:
            data = json.load(reply)["protocolSection"]
    except urllib.error.HTTPError as e:
        raise HTTPException(404 if e.code == 404 else 502, f"ClinicalTrials.gov could not provide {nct_id} (status {e.code}).")
    except (urllib.error.URLError, TimeoutError):
        raise HTTPException(502, "Could not reach ClinicalTrials.gov. Check the connection, or paste the criteria instead.")
    text = data.get("eligibilityModule", {}).get("eligibilityCriteria", "")
    try:
        criteria = parse_criteria(text)
    except ValueError:
        raise HTTPException(422, "This trial's criteria aren't in the usual format. Please paste them instead.")
    return {"title": data.get("identificationModule", {}).get("briefTitle", ""), "criteria": criteria}


@app.post("/screen")
def screen(request: ScreenRequest):
    require_model()
    parts, numbered = prepare_note(request.note)
    return start({"kind": "screen", "sentences": parts, "numbered": numbered, "criteria": [c.model_dump() for c in request.criteria],
                  "mode": request.mode, "k": request.k, "results": [], "aggregate": None}, run_screen)


@app.post("/benchmark")
def benchmark(request: BenchmarkRequest):
    require_model()
    parts, numbered = prepare_note(request.note)
    return start({"kind": "benchmark", "numbered": numbered, "criterion": request.criterion.model_dump(), "modes": request.modes, "k": request.k, "rows": []}, run_benchmark)


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "That job doesn't exist. It may have been lost when the app restarted.")
    return jobs[job_id]


def report_markdown(job: dict) -> str:
    lines = ["# Screening report", "", f"**{DISCLAIMER}**", "", f"- Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
             f"- Screening run: {job['created']}", f"- Model: {TARGET}", f"- Adapter: {TARGET_ADAPTER}", f"- Decoding: {job['mode']}" + (f" (k={job['k']}, draft {DRAFT}, {DRAFT_ADAPTER})" if job["mode"] == "speculative" else ""),
             "- Greedy decoding, no sampling", f"- Input SHA-256: {hashlib.sha256((job['numbered'] + json.dumps(job['criteria'], sort_keys=True)).encode()).hexdigest()}", "",
             f"## Overall preliminary outcome: {job['aggregate']}", "", RULE, "", "## Patient note (numbered as the model saw it)", "", "```", job["numbered"], "```", "", "## Criteria", ""]
    for i, r in enumerate(job["results"], 1):
        lines += [f"### {i}. ({r['criterion_type']}) {r['criterion']}", "", f"- Label: **{r['pred_label']}**", f"- Cited sentences: {r['pred_ids'] or 'none'}"]
        lines += [f"  - \"{text}\"" for text in r["evidence_text"]]
        lines += [f"- Explanation: {r['explanation']}", ""]
    return "\n".join(lines)


@app.get("/report/{job_id}")
def report(job_id: str, fmt: Literal["md", "html"] = "html"):
    job = jobs.get(job_id)
    if not job or job["kind"] != "screen" or job["status"] != "done":
        raise HTTPException(404, "That report isn't ready. Finish a screening run first.")
    text = report_markdown(job)
    if fmt == "md":
        return Response(text, media_type="text/markdown", headers={"Content-Disposition": "attachment; filename=screening_report.md"})
    page_html = f"<!doctype html><meta charset=utf-8><title>Screening report</title><pre style='white-space:pre-wrap;font:15px/1.5 system-ui;max-width:800px;margin:2em auto'>{html.escape(text)}</pre>"
    return Response(page_html, media_type="text/html", headers={"Content-Disposition": "attachment; filename=screening_report.html"})

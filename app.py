from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import io
import zipfile

import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

import analysis  # runtime temporal analysis engine

APP_ROOT = Path(__file__).resolve().parent
load_dotenv(APP_ROOT / ".env")

GPT_API_KEY = os.getenv("GPT_API_KEY")
openai_client = OpenAI(api_key=GPT_API_KEY) if GPT_API_KEY else None
SAMPLES_DIR = APP_ROOT / "samples"

app = FastAPI(title="Agricultural Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    context: str | None = None


def clamp_history(history: list[ChatMessage], limit: int = 12) -> list[ChatMessage]:
    if len(history) <= limit:
        return history
    return history[-limit:]


def compact_context(context: str | None, limit: int = 2000) -> str | None:
    if not context:
        return None
    if len(context) <= limit:
        return context
    return f"{context[:limit]}\n…(truncated)"


# ─────────────────────────────────────────────────────────────────
# File serving (unchanged)
# ─────────────────────────────────────────────────────────────────

def ensure_sample_path(relative_path: str) -> Path:
    resolved_path = (SAMPLES_DIR / relative_path).resolve()
    if resolved_path != SAMPLES_DIR and SAMPLES_DIR not in resolved_path.parents:
        raise HTTPException(status_code=400, detail="Invalid sample path")
    return resolved_path


def guess_content_type(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        return "application/json"
    if suffix == ".geojson":
        return "application/geo+json"
    if suffix == ".csv":
        return "text/csv"
    if suffix in {".txt", ".cpg", ".prj", ".qmd"}:
        return "text/plain"
    return "application/octet-stream"


@app.get("/samples")
def list_samples() -> JSONResponse:
    if not SAMPLES_DIR.exists():
        raise HTTPException(status_code=500, detail="Samples directory not found")
    entries: list[dict[str, Any]] = []
    for path in sorted(SAMPLES_DIR.rglob("*")):
        relative = path.relative_to(SAMPLES_DIR).as_posix()
        if path.is_dir():
            entries.append({"path": f"{relative}/", "size": 0, "kind": "directory"})
        else:
            entries.append({"path": relative, "size": path.stat().st_size, "kind": "file"})
    return JSONResponse({"root": str(SAMPLES_DIR), "files": entries})


@app.get("/samples/{file_path:path}")
def read_sample(file_path: str) -> FileResponse:
    target = ensure_sample_path(file_path)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Sample file not found")
    return FileResponse(target, media_type=guess_content_type(target))


# ─────────────────────────────────────────────────────────────────
# Temporal Analysis endpoints  (all compute at runtime)
# ─────────────────────────────────────────────────────────────────

@app.get("/temporal/stability")
def temporal_stability() -> JSONResponse:
    """4A – CV-based genotype stability classification."""
    return JSONResponse(analysis.get_stability())


@app.get("/temporal/yield-class")
def temporal_yield_class() -> JSONResponse:
    """4B – Yield class distribution (tertiles on stable genotypes)."""
    return JSONResponse(analysis.get_yield_class())


@app.get("/temporal/tukey")
def temporal_tukey() -> JSONResponse:
    """4D – ANOVA significance + significant pairs per feature."""
    return JSONResponse(analysis.get_tukey())


@app.get("/temporal/correlation")
def temporal_correlation() -> JSONResponse:
    """4E – Pearson/Spearman correlation of each feature mean with Yield."""
    return JSONResponse(analysis.get_correlation())


@app.get("/temporal/growth-senescence")
def temporal_growth_senescence() -> JSONResponse:
    """4F – Growth and senescence rates by yield class."""
    return JSONResponse(analysis.get_growth_senescence())


@app.get("/temporal/phenology")
def temporal_phenology() -> JSONResponse:
    """4G – Phenology metrics (peak, time_to_peak, senescence_duration, staygreen_auc)."""
    return JSONResponse(analysis.get_phenology())


@app.get("/temporal/interpretation")
def temporal_interpretation() -> JSONResponse:
    """4I – Biological interpretation of significant features."""
    return JSONResponse(analysis.get_feature_interpretation())


@app.get("/temporal/outliers")
def temporal_outliers(
    yield_class: list[str] | None = Query(default=None),
) -> JSONResponse:
    """4K – Genotype outlier Z-scores. Filter by yield_class if provided."""
    return JSONResponse(analysis.get_outliers(yield_class))


@app.get("/temporal/category-summary")
def temporal_category_summary() -> JSONResponse:
    """4L – Category-level outlier summary + heatmap matrix."""
    return JSONResponse(analysis.get_category_summary())


@app.get("/temporal/chat-context")
def temporal_chat_context() -> JSONResponse:
    """Compact temporal summary for chat prompts."""
    return JSONResponse(analysis.get_chat_context())


@app.get("/temporal/export")
def temporal_export() -> StreamingResponse:
    """Export all temporal analysis tables as a zip of CSVs."""
    stability = analysis.get_stability()
    yield_class = analysis.get_yield_class()
    tukey = analysis.get_tukey()
    correlation = analysis.get_correlation()
    phenology = analysis.get_phenology()
    interpretation = analysis.get_feature_interpretation()
    outliers = analysis.get_outliers()
    category_summary = analysis.get_category_summary()

    csv_sets: dict[str, list[dict[str, Any]]] = {
        "stability_summary.csv": stability.get("summary", []),
        "stability_category_counts.csv": stability.get("category_counts", []),
        "yield_class_distribution.csv": yield_class.get("distribution", []),
        "yield_class_genotype_table.csv": yield_class.get("genotype_table", []),
        "tukey_significant_features.csv": tukey.get("tukey", []),
        "feature_correlations.csv": correlation.get("correlations", []),
        "phenology_records.csv": phenology.get("records", []),
        "phenology_features.csv": [{"feature": f} for f in phenology.get("features", [])],
        "phenology_metrics.csv": [{"metric": m} for m in phenology.get("metrics", [])],
        "feature_interpretation.csv": interpretation.get("interpretations", []),
        "outliers.csv": outliers.get("outliers", []),
        "outlier_heatmap.csv": outliers.get("heatmap", []),
        "outlier_yield_classes.csv": [{"yield_class": c} for c in outliers.get("available_yield_classes", [])],
        "category_summary.csv": category_summary.get("category_summary", []),
        "category_heatmap.csv": category_summary.get("heatmap_matrix", []),
    }

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for name, records in csv_sets.items():
            frame = pd.DataFrame(records)
            csv_text = frame.to_csv(index=False)
            zip_file.writestr(name, csv_text)

    buffer.seek(0)
    headers = {"Content-Disposition": "attachment; filename=temporal_export.zip"}
    return StreamingResponse(buffer, media_type="application/zip", headers=headers)


# ─────────────────────────────────────────────────────────────────
# Chat endpoint (GPT-4o mini)
# ─────────────────────────────────────────────────────────────────

@app.post("/chat")
def chat(req: ChatRequest) -> JSONResponse:
    if not openai_client:
        raise HTTPException(status_code=500, detail="GPT_API_KEY is not set")

    context = compact_context(req.context)
    system_prompt = (
        "You are an agronomy assistant. Provide concise, evidence-based guidance. "
        "If you lack data, state the limitation and ask for what is needed."
    )
    if context:
        system_prompt += "\n\nContext data:\n" + context

    history = clamp_history(req.history)
    messages = [{"role": "system", "content": system_prompt}]
    messages += [{"role": m.role, "content": m.content} for m in history]
    messages.append({"role": "user", "content": req.message})

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=500,
    )
    reply = response.choices[0].message.content or ""
    return JSONResponse({"reply": reply.strip()})

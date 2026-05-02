from __future__ import annotations

import io
import os
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

import analysis  # runtime temporal analysis engine

APP_ROOT = Path(__file__).resolve().parent
load_dotenv(APP_ROOT / ".env")

GPT_API_KEY = os.getenv("GPT_API_KEY")
openai_client = OpenAI(api_key=GPT_API_KEY) if GPT_API_KEY else None
SAMPLES_DIR = APP_ROOT / "samples"

# In-memory session store: session_id -> {"temporal_csv": Path, "timestamps_dir": Path, "geojson": Path}
_sessions: dict[str, dict[str, Any]] = {}

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
# File serving
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
# VI computation from pixel-level CSVs
# ─────────────────────────────────────────────────────────────────

VI_FORMULAS = {
    "NDRE":   lambda d: (d["NIR"] - d["Rededge"]) / (d["NIR"] + d["Rededge"]),
    "MTCI":   lambda d: (d["NIR"] - d["Rededge"]) / (d["Rededge"] - d["Red"]),
    "NDVI":   lambda d: (d["NIR"] - d["Red"]) / (d["NIR"] + d["Red"]),
    "NDWI":   lambda d: (d["NIR"] - d["Green"]) / (d["NIR"] + d["Green"]),
    "MSI":    lambda d: d["Red"] / d["NIR"],
    "WI":     lambda d: d["NIR"] / d["Green"],
    "SAVI":   lambda d: ((d["NIR"] - d["Red"]) / (d["NIR"] + d["Red"] + 0.5)) * 1.5,
    "EVI":    lambda d: 2.5 * (d["NIR"] - d["Red"]) / (d["NIR"] + 6 * d["Red"] - 7.5 * d["Blue"] + 1),
    "EXG":    lambda d: 2 * d["Green"] - d["Red"] - d["Blue"],
    "CIgreen":lambda d: (d["NIR"] / d["Green"]) - 1,
    "NDCI":   lambda d: (d["Rededge"] - d["Red"]) / (d["Rededge"] + d["Red"]),
    "PSRI":   lambda d: (d["Red"] - d["Green"]) / d["NIR"],
    "SIPI":   lambda d: (d["NIR"] - d["Blue"]) / (d["NIR"] + d["Red"]),
}

VI_COLS = list(VI_FORMULAS.keys())
STAT_AGG = ["sum", "count", "mean", "median", "std", "var", "min", "max"]


def compute_vi_stats(pixel_csv_path: Path) -> pd.DataFrame:
    """Read pixel CSV, compute 13 VIs, group by PLOT_ID, return stats df."""
    df = pd.read_csv(pixel_csv_path)
    df.columns = df.columns.str.strip()

    required = {"NIR", "Rededge", "Red", "Blue", "Green", "PLOT_ID"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Pixel CSV missing columns: {missing}")

    for vi, fn in VI_FORMULAS.items():
        df[vi] = fn(df)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    summary = df.groupby("PLOT_ID")[VI_COLS].agg(STAT_AGG).reset_index()
    summary.columns = [
        "_".join(col).strip() if col[1] else col[0]
        for col in summary.columns.values
    ]
    return summary


def process_reflectance_pair(rgb_path: Path, nir_path: Path, band_mapping: dict,
                              shapefile_geojson: dict, timestamp_label: str,
                              out_dir: Path) -> Path:
    """
    Placeholder: In production this would extract per-pixel band values using
    rasterio + shapely from the uploaded reflectance maps and shapefile.
    For now we produce a stub STATS CSV that the analysis engine can consume.
    Returns path to the STATS CSV.
    """
    # Derive genotypes from shapefile properties
    rows = []
    for feat in shapefile_geojson.get("features", []):
        props = feat.get("properties", {})
        plot_id = str(props.get("PLOT_ID", props.get("plot_id", "")))
        # Genotype is last segment of PLOT_ID separated by "_"
        parts = plot_id.split("_")
        genotype = parts[-1] if parts else "0"
        rows.append({"PLOT_ID": plot_id, "genotype": genotype})

    df = pd.DataFrame(rows)
    for vi in VI_COLS:
        for stat in STAT_AGG:
            df[f"{vi}_{stat}"] = np.nan

    out_path = out_dir / f"{timestamp_label}_STATS.csv"
    df.to_csv(out_path, index=False)
    return out_path


# ─────────────────────────────────────────────────────────────────
# Upload: Mode A – Reflectance maps + shapefile (12 timestamps)
# ─────────────────────────────────────────────────────────────────

@app.post("/upload/reflectance-maps")
async def upload_reflectance_maps(
    rgb_images: list[UploadFile] = File(..., description="12 RGB reflectance maps"),
    nir_images: list[UploadFile] = File(..., description="12 NIR reflectance maps"),
    shapefile_json: UploadFile = File(..., description="Shapefile as GeoJSON"),
    band_red: int = Form(1),
    band_green: int = Form(2),
    band_blue: int = Form(3),
    band_nir: int = Form(1),
    band_rededge: int = Form(2),
    timestamps: str = Form("", description="Comma-separated timestamp labels"),
) -> JSONResponse:
    """
    Accept 12 RGB + 12 NIR reflectance maps and a shapefile (GeoJSON).
    Extract per-plot pixel values, compute 13 vegetation indices,
    aggregate stats per plot per timestamp, store in session.
    """
    if len(rgb_images) != len(nir_images):
        raise HTTPException(400, "RGB and NIR image counts must match.")
    if len(rgb_images) == 0:
        raise HTTPException(400, "At least one image pair required.")

    session_id = str(uuid.uuid4())
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"session_{session_id}_"))
    ts_dir = tmp_dir / "timestamps"
    ts_dir.mkdir()

    import json as json_module

    # Parse GeoJSON
    geojson_bytes = await shapefile_json.read()
    try:
        geojson_data = json_module.loads(geojson_bytes.decode("utf-8"))
    except Exception as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(400, f"Invalid GeoJSON: {exc}")

    # Parse timestamp labels
    ts_labels = [t.strip() for t in timestamps.split(",") if t.strip()]
    if not ts_labels:
        ts_labels = [f"t{i+1}" for i in range(len(rgb_images))]
    if len(ts_labels) < len(rgb_images):
        ts_labels += [f"t{i+1}" for i in range(len(ts_labels), len(rgb_images))]

    band_info = {
        "band_red": band_red, "band_green": band_green, "band_blue": band_blue,
        "band_nir": band_nir, "band_rededge": band_rededge,
    }

    stats_files: list[str] = []
    for i, (rgb_file, nir_file) in enumerate(zip(rgb_images, nir_images)):
        label = ts_labels[i]
        rgb_path = tmp_dir / f"rgb_{i}.tif"
        nir_path = tmp_dir / f"nir_{i}.tif"

        content_rgb = await rgb_file.read()
        content_nir = await nir_file.read()
        rgb_path.write_bytes(content_rgb)
        nir_path.write_bytes(content_nir)

        stats_df = process_reflectance_pair(
            rgb_path, nir_path, band_info, geojson_data, label, ts_dir
        )
        stats_files.append(stats_df.name if isinstance(stats_df, Path) else str(stats_df))

    # Build a temporal CSV from genotype info in GeoJSON
    rows = []
    for feat in geojson_data.get("features", []):
        props = feat.get("properties", {})
        plot_id = str(props.get("PLOT_ID", props.get("plot_id", "")))
        parts = plot_id.split("_")
        genotype = parts[-1] if parts else "0"
        rows.append({
            "PLOT_ID": plot_id,
            "genotype": genotype,
            "Yield": props.get("Yield", np.nan),
        })

    temporal_df = pd.DataFrame(rows)
    temporal_path = tmp_dir / "temporalDataSet.csv"
    temporal_df.to_csv(temporal_path, index=False)

    # Save GeoJSON for map
    geojson_path = tmp_dir / "plots.geojson"
    geojson_path.write_text(json_module.dumps(geojson_data))

    _sessions[session_id] = {
        "temporal_csv": temporal_path,
        "timestamps_dir": ts_dir,
        "geojson": geojson_path,
        "mode": "reflectance",
        "timestamp_count": len(rgb_images),
        "timestamp_labels": ts_labels,
    }

    # Invalidate analysis cache for this session
    analysis.invalidate_session_cache(session_id)

    return JSONResponse({
        "session_id": session_id,
        "timestamp_count": len(rgb_images),
        "timestamp_labels": ts_labels,
        "message": "Reflectance maps processed. Use session_id in analysis requests.",
    })


# ─────────────────────────────────────────────────────────────────
# Upload: Mode B – Temporal feature CSV + per-timestamp STATS CSVs
# ─────────────────────────────────────────────────────────────────

@app.post("/upload/temporal-csvs")
async def upload_temporal_csvs(
    temporal_csv: UploadFile = File(..., description="Main temporal feature CSV"),
    timestamp_csvs: list[UploadFile] = File(..., description="Per-timestamp pixel CSVs (up to 12)"),
    shapefile_json: UploadFile | None = File(default=None, description="Optional GeoJSON shapefile"),
) -> JSONResponse:
    """
    Accept a main temporal CSV (must have PLOT_ID, genotype columns) and
    up to 12 per-timestamp pixel-level CSVs (with NIR/Red/Green/Blue/Rededge/PLOT_ID).
    Computes VI STATS for each timestamp and stores in a session.
    """
    session_id = str(uuid.uuid4())
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"session_{session_id}_"))
    ts_dir = tmp_dir / "timestamps"
    ts_dir.mkdir()

    import json as json_module

    # Save + validate temporal CSV
    temporal_bytes = await temporal_csv.read()
    try:
        temporal_df = pd.read_csv(io.BytesIO(temporal_bytes))
        temporal_df.columns = temporal_df.columns.str.strip()
    except Exception as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(400, f"Could not parse temporal CSV: {exc}")

    required_cols = {"PLOT_ID", "genotype"}
    if not required_cols.issubset(set(temporal_df.columns)):
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(400, f"Temporal CSV must have columns: {required_cols}. Found: {set(temporal_df.columns)}")

    temporal_path = tmp_dir / "temporalDataSet.csv"
    temporal_path.write_bytes(temporal_bytes)

    # Process each timestamp pixel CSV → STATS CSV
    processed_labels: list[str] = []
    errors: list[str] = []

    for i, ts_file in enumerate(timestamp_csvs):
        label = f"t{i+1}"
        try:
            ts_bytes = await ts_file.read()
            pixel_path = tmp_dir / f"pixels_{i}.csv"
            pixel_path.write_bytes(ts_bytes)
            stats_df = compute_vi_stats(pixel_path)
            out_name = f"final_{label}_STATS.csv"
            stats_df.to_csv(ts_dir / out_name, index=False)
            processed_labels.append(label)
        except Exception as exc:
            errors.append(f"Timestamp {i+1} ({ts_file.filename}): {exc}")

    # Handle optional shapefile
    geojson_path: Path | None = None
    if shapefile_json is not None:
        geojson_bytes = await shapefile_json.read()
        try:
            geojson_data = json_module.loads(geojson_bytes.decode("utf-8"))
            geojson_path = tmp_dir / "plots.geojson"
            geojson_path.write_text(json_module.dumps(geojson_data))
        except Exception:
            geojson_path = None

    _sessions[session_id] = {
        "temporal_csv": temporal_path,
        "timestamps_dir": ts_dir,
        "geojson": geojson_path,
        "mode": "temporal-csvs",
        "timestamp_count": len(processed_labels),
        "timestamp_labels": processed_labels,
    }

    analysis.invalidate_session_cache(session_id)

    return JSONResponse({
        "session_id": session_id,
        "processed_timestamps": len(processed_labels),
        "timestamp_labels": processed_labels,
        "errors": errors,
        "has_shapefile": geojson_path is not None,
        "message": "Temporal CSVs processed. Use session_id in analysis requests.",
    })


# ─────────────────────────────────────────────────────────────────
# Upload: Mode C – Single temporal feature CSV only (simplest case)
# ─────────────────────────────────────────────────────────────────

@app.post("/upload/temporal-csv-only")
async def upload_temporal_csv_only(
    temporal_csv: UploadFile = File(..., description="Main temporal feature CSV"),
    shapefile_json: UploadFile | None = File(default=None, description="Optional GeoJSON shapefile"),
) -> JSONResponse:
    """
    Simplest upload: a single temporal CSV that already contains PLOT_ID, genotype,
    and any combination of VI features / Yield.
    - If Yield is present → used directly for analysis and to evaluate predictions.
    - If Yield is absent  → SVR model (svr_combined.pkl) will predict it.
    No per-timestamp pixel CSVs are required.
    """
    import json as json_module

    session_id = str(uuid.uuid4())
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"session_{session_id}_"))
    ts_dir = tmp_dir / "timestamps"
    ts_dir.mkdir()

    temporal_bytes = await temporal_csv.read()
    try:
        temporal_df = pd.read_csv(io.BytesIO(temporal_bytes))
        temporal_df.columns = temporal_df.columns.str.strip()
    except Exception as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(400, f"Could not parse CSV: {exc}")

    required_cols = {"PLOT_ID", "genotype"}
    if not required_cols.issubset(set(temporal_df.columns)):
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(
            400,
            f"CSV must have columns: {required_cols}. Found: {set(temporal_df.columns)}",
        )

    has_yield = "Yield" in temporal_df.columns

    # If Yield is missing, predict it using the SVR model
    if not has_yield:
        svr_path = APP_ROOT / "svr_combined.pkl"
        if svr_path.exists():
            import pickle as _pickle
            try:
                with open(svr_path, "rb") as fh:
                    svr_model = _pickle.load(fh)
                exclude = {"PLOT_ID", "genotype", "experiment", "Yield"}
                feat_cols = [
                    c for c in temporal_df.columns
                    if c not in exclude and pd.api.types.is_numeric_dtype(temporal_df[c])
                ]
                if feat_cols and hasattr(svr_model, "feature_names_in_"):
                    X = temporal_df[feat_cols].reindex(
                        columns=list(svr_model.feature_names_in_), fill_value=0
                    )
                    temporal_df["Yield"] = svr_model.predict(X.values)
                    has_yield = True
            except Exception:
                pass  # leave Yield absent — analyses will degrade gracefully

    temporal_path = tmp_dir / "temporalDataSet.csv"
    temporal_df.to_csv(temporal_path, index=False)

    # Optional shapefile
    geojson_path: Path | None = None
    if shapefile_json is not None:
        geojson_bytes = await shapefile_json.read()
        try:
            geojson_data = json_module.loads(geojson_bytes.decode("utf-8"))
            geojson_path = tmp_dir / "plots.geojson"
            geojson_path.write_text(json_module.dumps(geojson_data))
        except Exception:
            geojson_path = None

    _sessions[session_id] = {
        "temporal_csv": temporal_path,
        "timestamps_dir": ts_dir,
        "geojson": geojson_path,
        "mode": "temporal-csv-only",
        "timestamp_count": 0,
        "timestamp_labels": [],
        "has_yield": has_yield,
    }

    analysis.invalidate_session_cache(session_id)

    return JSONResponse({
        "session_id": session_id,
        "has_yield": has_yield,
        "has_shapefile": geojson_path is not None,
        "row_count": len(temporal_df),
        "column_count": len(temporal_df.columns),
        "message": (
            "CSV uploaded. Yield column found — predictions will use actual values."
            if has_yield
            else "CSV uploaded. No Yield column — SVR model will predict yield."
        ),
    })


# ─────────────────────────────────────────────────────────────────
# Session GeoJSON endpoint
# ─────────────────────────────────────────────────────────────────

@app.get("/session/{session_id}/geojson", response_model=None)
def session_geojson(session_id: str) -> FileResponse | JSONResponse:
    sess = _sessions.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    gj = sess.get("geojson")
    if not gj or not Path(gj).exists():
        raise HTTPException(404, "No shapefile uploaded for this session")
    return FileResponse(gj, media_type="application/geo+json")


@app.get("/session/{session_id}/info")
def session_info(session_id: str) -> JSONResponse:
    sess = _sessions.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    return JSONResponse({
        "session_id": session_id,
        "mode": sess.get("mode"),
        "timestamp_count": sess.get("timestamp_count", 0),
        "timestamp_labels": sess.get("timestamp_labels", []),
        "has_shapefile": sess.get("geojson") is not None,
    })


# ─────────────────────────────────────────────────────────────────
# Helper: resolve analysis engine for session or default
# ─────────────────────────────────────────────────────────────────

def _resolve_engine(session_id: str | None) -> analysis.AnalysisEngine:
    if session_id and session_id in _sessions:
        sess = _sessions[session_id]
        return analysis.AnalysisEngine(
            temporal_csv=sess["temporal_csv"],
            timestamps_dir=sess["timestamps_dir"],
            session_id=session_id,
        )
    return analysis.default_engine


# ─────────────────────────────────────────────────────────────────
# Temporal Analysis endpoints
# ─────────────────────────────────────────────────────────────────

@app.get("/temporal/stability")
def temporal_stability(session_id: str | None = Query(default=None)) -> JSONResponse:
    return JSONResponse(_resolve_engine(session_id).get_stability())


@app.get("/temporal/yield-class")
def temporal_yield_class(session_id: str | None = Query(default=None)) -> JSONResponse:
    return JSONResponse(_resolve_engine(session_id).get_yield_class())


@app.get("/temporal/tukey")
def temporal_tukey(session_id: str | None = Query(default=None)) -> JSONResponse:
    return JSONResponse(_resolve_engine(session_id).get_tukey())


@app.get("/temporal/correlation")
def temporal_correlation(session_id: str | None = Query(default=None)) -> JSONResponse:
    return JSONResponse(_resolve_engine(session_id).get_correlation())


@app.get("/temporal/growth-senescence")
def temporal_growth_senescence(session_id: str | None = Query(default=None)) -> JSONResponse:
    return JSONResponse(_resolve_engine(session_id).get_growth_senescence())


@app.get("/temporal/phenology")
def temporal_phenology(session_id: str | None = Query(default=None)) -> JSONResponse:
    return JSONResponse(_resolve_engine(session_id).get_phenology())


@app.get("/temporal/interpretation")
def temporal_interpretation(session_id: str | None = Query(default=None)) -> JSONResponse:
    return JSONResponse(_resolve_engine(session_id).get_feature_interpretation())


@app.get("/temporal/outliers")
def temporal_outliers(
    yield_class: list[str] | None = Query(default=None),
    session_id: str | None = Query(default=None),
) -> JSONResponse:
    return JSONResponse(_resolve_engine(session_id).get_outliers(yield_class))


@app.get("/temporal/category-summary")
def temporal_category_summary(session_id: str | None = Query(default=None)) -> JSONResponse:
    return JSONResponse(_resolve_engine(session_id).get_category_summary())


@app.get("/temporal/chat-context")
def temporal_chat_context(session_id: str | None = Query(default=None)) -> JSONResponse:
    return JSONResponse(_resolve_engine(session_id).get_chat_context())


@app.get("/temporal/export")
def temporal_export(session_id: str | None = Query(default=None)) -> StreamingResponse:
    engine = _resolve_engine(session_id)
    stability = engine.get_stability()
    yield_class = engine.get_yield_class()
    tukey = engine.get_tukey()
    correlation = engine.get_correlation()
    phenology = engine.get_phenology()
    interpretation = engine.get_feature_interpretation()
    outliers = engine.get_outliers()
    category_summary = engine.get_category_summary()

    csv_sets: dict[str, list[dict[str, Any]]] = {
        "stability_summary.csv": stability.get("summary", []),
        "stability_category_counts.csv": stability.get("category_counts", []),
        "yield_class_distribution.csv": yield_class.get("distribution", []),
        "yield_class_genotype_table.csv": yield_class.get("genotype_table", []),
        "tukey_significant_features.csv": tukey.get("tukey", []),
        "feature_correlations.csv": correlation.get("correlations", []),
        "phenology_records.csv": phenology.get("records", []),
        "feature_interpretation.csv": interpretation.get("interpretations", []),
        "outliers.csv": outliers.get("outliers", []),
        "outlier_heatmap.csv": outliers.get("heatmap", []),
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
# Yield Prediction endpoint (LMM-style)
# ─────────────────────────────────────────────────────────────────

@app.get("/yield-prediction")
def yield_prediction(session_id: str | None = Query(default=None)) -> JSONResponse:
    """
    Simple yield prediction using mean VI values as features.
    Returns per-genotype predicted yield and feature importances.
    """
    engine = _resolve_engine(session_id)
    try:
        result = engine.get_yield_prediction()
        return JSONResponse(result)
    except Exception as exc:
        raise HTTPException(500, f"Yield prediction failed: {exc}")


# ─────────────────────────────────────────────────────────────────
# Chat endpoint
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

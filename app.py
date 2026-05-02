from __future__ import annotations

import io
import os
import shutil
import tempfile
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import jwt
import boto3
from botocore.exceptions import ClientError
from passlib.context import CryptContext
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from bson import ObjectId

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

import analysis  # runtime temporal analysis engine

APP_ROOT = Path(__file__).resolve().parent
load_dotenv(APP_ROOT / ".env")

GPT_API_KEY = os.getenv("GPT_API_KEY")
openai_client = OpenAI(api_key=GPT_API_KEY) if GPT_API_KEY else None
SAMPLES_DIR = APP_ROOT / "samples"

MONGO_URL = os.getenv("MongoURl")
MONGO_DB = os.getenv("MONGO_DB", "genosence")
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")

R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")

mongo_client = MongoClient(MONGO_URL) if MONGO_URL else None
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
MAX_BCRYPT_PASSWORD_BYTES = 72

# In-memory session store: session_id -> {"temporal_csv": Path, "timestamps_dir": Path, "geojson": Path}
_sessions: dict[str, dict[str, Any]] = {}

app = FastAPI(title="Agricultural Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def ensure_indexes() -> None:
    if not mongo_client:
        return
    users = get_users_collection()
    users.create_index("email", unique=True)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    context: str | None = None


class SignUpRequest(BaseModel):
    name: str
    email: str
    password: str


class SignInRequest(BaseModel):
    email: str
    password: str


class UserPublic(BaseModel):
    id: str
    name: str
    email: str
    created_at: str
    last_processed_files: dict[str, Any] | None = None


def get_db():
    if not mongo_client:
        raise HTTPException(status_code=500, detail="MongoDB is not configured")
    return mongo_client[MONGO_DB]


def get_users_collection():
    return get_db()["users"]


def get_temporal_collection():
    return get_db()["temporal_features"]


def get_timestamps_collection():
    return get_db()["timestamps"]


def get_shapefiles_collection():
    return get_db()["shapefiles"]


def hash_password(password: str) -> str:
    if len(password.encode("utf-8")) > MAX_BCRYPT_PASSWORD_BYTES:
        raise HTTPException(status_code=400, detail="Password must be 72 bytes or fewer")
    return pwd_context.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    if len(password.encode("utf-8")) > MAX_BCRYPT_PASSWORD_BYTES:
        return False
    return pwd_context.verify(password, hashed)


def create_access_token(user_id: str, email: str) -> str:
    if not JWT_SECRET:
        raise HTTPException(status_code=500, detail="JWT_SECRET is not configured")
    expires = datetime.now(tz=timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {"sub": user_id, "email": email, "exp": expires}
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def decode_access_token(token: str) -> dict[str, Any]:
    if not JWT_SECRET:
        raise HTTPException(status_code=500, detail="JWT_SECRET is not configured")
    return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])


def get_current_user(request: Request) -> dict[str, Any] | None:
    token = request.cookies.get("access_token")
    if not token:
        return None
    try:
        payload = decode_access_token(token)
        user_id = payload.get("sub")
        if not user_id:
            return None
        users = get_users_collection()
        user = users.find_one({"_id": ObjectId(user_id)})
        return user
    except Exception:
        return None


def to_public_user(user: dict[str, Any]) -> UserPublic:
    return UserPublic(
        id=str(user["_id"]),
        name=user.get("name", ""),
        email=user.get("email", ""),
        created_at=user.get("created_at", ""),
        last_processed_files=user.get("last_processed_files"),
    )


def get_r2_client():
    if not (R2_ACCOUNT_ID and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY):
        return None
    endpoint = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
    )


def ensure_user_r2_prefix(user_id: str) -> None:
    client = get_r2_client()
    if not client or not R2_BUCKET_NAME:
        return
    key = f"uploads/{user_id}/.keep"
    try:
        client.put_object(Bucket=R2_BUCKET_NAME, Key=key, Body=b"")
    except ClientError:
        pass


def upload_bytes_to_r2(user_id: str, content: bytes, filename: str) -> str | None:
    client = get_r2_client()
    if not client or not R2_BUCKET_NAME:
        return None
    ext = Path(filename).suffix.lower()
    key = f"uploads/{user_id}/{uuid.uuid4().hex}{ext}"
    client.put_object(Bucket=R2_BUCKET_NAME, Key=key, Body=content)
    return key


def clamp_history(history: list[ChatMessage], limit: int = 12) -> list[ChatMessage]:
    if len(history) <= limit:
        return history
    return history[-limit:]


def compact_context(context: str | None, limit: int = 2000) -> str | None:
    if not context:
        return None
    if len(context) <= limit:
        return context
    return f"{context[:limit]}\n...(truncated)"


# ─────────────────────────────────────────────────────────────────
# Auth endpoints
# ─────────────────────────────────────────────────────────────────

@app.post("/auth/signup")
def auth_signup(payload: SignUpRequest) -> JSONResponse:
    users = get_users_collection()
    email = payload.email.strip().lower()
    if not email or not payload.password:
        raise HTTPException(status_code=400, detail="Email and password required")

    user_doc = {
        "name": payload.name.strip(),
        "email": email,
        "password_hash": hash_password(payload.password),
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "last_processed_files": None,
    }

    try:
        result = users.insert_one(user_doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="Email already registered")

    user_id = str(result.inserted_id)
    ensure_user_r2_prefix(user_id)
    token = create_access_token(user_id, email)
    user = users.find_one({"_id": ObjectId(user_id)})
    response = JSONResponse({"user": to_public_user(user).model_dump()})
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=JWT_EXPIRE_MINUTES * 60,
    )
    return response


@app.post("/auth/signin")
def auth_signin(payload: SignInRequest) -> JSONResponse:
    users = get_users_collection()
    email = payload.email.strip().lower()
    user = users.find_one({"email": email})
    if not user or not verify_password(payload.password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(str(user["_id"]), email)
    response = JSONResponse({"user": to_public_user(user).model_dump()})
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=JWT_EXPIRE_MINUTES * 60,
    )
    return response


@app.post("/auth/signout")
def auth_signout() -> JSONResponse:
    response = JSONResponse({"ok": True})
    response.delete_cookie("access_token")
    return response


@app.get("/auth/me")
def auth_me(request: Request) -> JSONResponse:
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return JSONResponse({"user": to_public_user(user).model_dump()})


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


def delete_user_upload_data(user_id: str) -> None:
    get_temporal_collection().delete_many({"user_id": user_id})
    get_timestamps_collection().delete_many({"user_id": user_id})
    get_shapefiles_collection().delete_many({"user_id": user_id})


def store_temporal_features(user_id: str, session_id: str, temporal_df: pd.DataFrame) -> list[str]:
    collection = get_temporal_collection()
    records = temporal_df.to_dict(orient="records")
    if not records:
        return []
    doc = {
        "user_id": user_id,
        "session_id": session_id,
        "records": records,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    result = collection.insert_one(doc)
    return [str(result.inserted_id)]


def _extract_date_label(name: str) -> str:
    stem = Path(name).stem
    if stem.startswith("final_"):
        stem = stem[6:]
    if stem.endswith("_STATS"):
        stem = stem[:-6]
    return stem.replace("-", "_")


def store_timestamps(user_id: str, session_id: str, timestamps_dir: Path) -> list[str]:
    collection = get_timestamps_collection()
    timestamps: list[dict[str, Any]] = []
    for csv_path in sorted(timestamps_dir.glob("*.csv")):
        date_label = _extract_date_label(csv_path.name)
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        records = df.to_dict(orient="records")
        timestamps.append({
            "date": date_label,
            "records": records,
        })
    if not timestamps:
        return []
    doc = {
        "user_id": user_id,
        "session_id": session_id,
        "timestamps": timestamps,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    result = collection.insert_one(doc)
    return [str(result.inserted_id)]


def store_shapefile_geojson(
    user_id: str,
    session_id: str,
    geojson_data: dict[str, Any],
    filename: str | None,
) -> str:
    collection = get_shapefiles_collection()
    doc = {
        "user_id": user_id,
        "session_id": session_id,
        "filename": filename,
        "geojson": geojson_data,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    result = collection.insert_one(doc)
    return str(result.inserted_id)


def update_user_last_processed_files(
    user_id: str,
    temporal_key: str | None,
    shapefile_key: str | None,
    reflectance_keys: list[str],
    timestamp_ids: list[str],
    temporal_ids: list[str] | None = None,
    shapefile_id: str | None = None,
) -> None:
    users = get_users_collection()
    users.update_one(
        {"_id": ObjectId(user_id)},
        {
            "$set": {
                "last_processed_files": {
                    "temporal_feature_key": temporal_key,
                    "temporal_feature_ids": temporal_ids or [],
                    "shapefile_key": shapefile_key,
                    "shapefile_id": shapefile_id,
                    "reflectance_map_keys": reflectance_keys,
                    "timestamp_ids": timestamp_ids,
                }
            }
        },
    )


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
    request: Request,
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
        print(f"/upload/reflectance-maps: invalid GeoJSON: {exc}")
        raise HTTPException(400, f"Invalid GeoJSON: {exc}")
    print("/upload/reflectance-maps: GeoJSON parsed successfully")

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

    user = get_current_user(request)
    user_id = str(user["_id"]) if user else None
    reflectance_keys: list[str] = []
    if user_id:
        ensure_user_r2_prefix(user_id)
    else:
        print("/upload/reflectance-maps: no authenticated user; skipping R2 + Mongo writes")

    stats_files: list[str] = []
    for i, (rgb_file, nir_file) in enumerate(zip(rgb_images, nir_images)):
        label = ts_labels[i]
        rgb_path = tmp_dir / f"rgb_{i}.tif"
        nir_path = tmp_dir / f"nir_{i}.tif"

        content_rgb = await rgb_file.read()
        content_nir = await nir_file.read()
        rgb_path.write_bytes(content_rgb)
        nir_path.write_bytes(content_nir)

        if user_id:
            try:
                rgb_key = upload_bytes_to_r2(user_id, content_rgb, rgb_file.filename)
                nir_key = upload_bytes_to_r2(user_id, content_nir, nir_file.filename)
                if rgb_key:
                    reflectance_keys.append(rgb_key)
                    print(f"/upload/reflectance-maps: uploaded RGB {rgb_file.filename} -> {rgb_key}")
                if nir_key:
                    reflectance_keys.append(nir_key)
                    print(f"/upload/reflectance-maps: uploaded NIR {nir_file.filename} -> {nir_key}")
            except Exception as exc:
                print(f"/upload/reflectance-maps: failed to upload reflectance maps: {exc}")

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

    if user_id:
        shapefile_id = None
        temporal_ids: list[str] = []
        try:
            temporal_bytes = temporal_df.to_csv(index=False).encode("utf-8")
            temporal_key = upload_bytes_to_r2(user_id, temporal_bytes, "temporalDataSet.csv")
            print("/upload/reflectance-maps: uploaded temporalDataSet.csv to R2")
        except Exception:
            temporal_key = None
            print("/upload/reflectance-maps: failed to upload temporalDataSet.csv to R2")
        try:
            shapefile_key = upload_bytes_to_r2(user_id, geojson_bytes, shapefile_json.filename)
            print("/upload/reflectance-maps: uploaded shapefile GeoJSON to R2")
        except Exception:
            shapefile_key = None
            print("/upload/reflectance-maps: failed to upload shapefile GeoJSON to R2")
        try:
            delete_user_upload_data(user_id)
            temporal_ids = store_temporal_features(user_id, session_id, temporal_df)
            timestamp_ids = store_timestamps(user_id, session_id, ts_dir)
            shapefile_id = store_shapefile_geojson(user_id, session_id, geojson_data, shapefile_json.filename)
            print("/upload/reflectance-maps: stored temporal, timestamps, and shapefile in Mongo")
        except Exception as exc:
            timestamp_ids = []
            print(f"/upload/reflectance-maps: failed to store upload data in Mongo: {exc}")
        update_user_last_processed_files(
            user_id,
            temporal_key,
            shapefile_key,
            reflectance_keys,
            timestamp_ids,
            temporal_ids=temporal_ids,
            shapefile_id=shapefile_id,
        )
        print("/upload/reflectance-maps: updated user last_processed_files")

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
    request: Request,
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
    geojson_data: dict[str, Any] | None = None
    if shapefile_json is not None:
        geojson_bytes = await shapefile_json.read()
        try:
            geojson_data = json_module.loads(geojson_bytes.decode("utf-8"))
            geojson_path = tmp_dir / "plots.geojson"
            geojson_path.write_text(json_module.dumps(geojson_data))
        except Exception:
            geojson_path = None
            geojson_data = None

    _sessions[session_id] = {
        "temporal_csv": temporal_path,
        "timestamps_dir": ts_dir,
        "geojson": geojson_path,
        "mode": "temporal-csvs",
        "timestamp_count": len(processed_labels),
        "timestamp_labels": processed_labels,
    }

    analysis.invalidate_session_cache(session_id)

    user = get_current_user(request)
    if user:
        user_id = str(user["_id"])
        ensure_user_r2_prefix(user_id)
        shapefile_id = None
        temporal_ids: list[str] = []
        try:
            temporal_key = upload_bytes_to_r2(user_id, temporal_bytes, temporal_csv.filename)
        except Exception:
            temporal_key = None
        shapefile_key = None
        if shapefile_json is not None:
            try:
                shapefile_key = upload_bytes_to_r2(user_id, geojson_bytes, shapefile_json.filename)
            except Exception:
                shapefile_key = None
        try:
            delete_user_upload_data(user_id)
            temporal_ids = store_temporal_features(user_id, session_id, temporal_df)
            timestamp_ids = store_timestamps(user_id, session_id, ts_dir)
            if geojson_data is not None:
                shapefile_id = store_shapefile_geojson(user_id, session_id, geojson_data, shapefile_json.filename)
        except Exception:
            timestamp_ids = []
        update_user_last_processed_files(
            user_id,
            temporal_key,
            shapefile_key,
            [],
            timestamp_ids,
            temporal_ids=temporal_ids,
            shapefile_id=shapefile_id,
        )

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
    request: Request,
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
    geojson_data: dict[str, Any] | None = None
    if shapefile_json is not None:
        geojson_bytes = await shapefile_json.read()
        try:
            geojson_data = json_module.loads(geojson_bytes.decode("utf-8"))
            geojson_path = tmp_dir / "plots.geojson"
            geojson_path.write_text(json_module.dumps(geojson_data))
        except Exception:
            geojson_path = None
            geojson_data = None

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

    user = get_current_user(request)
    if user:
        user_id = str(user["_id"])
        ensure_user_r2_prefix(user_id)
        shapefile_id = None
        temporal_ids: list[str] = []
        try:
            temporal_key = upload_bytes_to_r2(user_id, temporal_bytes, temporal_csv.filename)
        except Exception:
            temporal_key = None
        shapefile_key = None
        if shapefile_json is not None:
            try:
                shapefile_key = upload_bytes_to_r2(user_id, geojson_bytes, shapefile_json.filename)
            except Exception:
                shapefile_key = None
        try:
            delete_user_upload_data(user_id)
            temporal_ids = store_temporal_features(user_id, session_id, temporal_df)
            if geojson_data is not None:
                shapefile_id = store_shapefile_geojson(user_id, session_id, geojson_data, shapefile_json.filename)
        except Exception:
            pass
        update_user_last_processed_files(
            user_id,
            temporal_key,
            shapefile_key,
            [],
            [],
            temporal_ids=temporal_ids,
            shapefile_id=shapefile_id,
        )

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

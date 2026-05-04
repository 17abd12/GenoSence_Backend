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

MONGO_URL = os.getenv("MONGO_URL") or os.getenv("MongoURl")
MONGO_DB = os.getenv("MONGO_DB", "genosence")
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
IS_PRODUCTION = os.getenv("ENVIRONMENT") == "production" or os.getenv("RAILWAY_ENVIRONMENT") is not None or "railway.app" in os.getenv("RAILWAY_PUBLIC_DOMAIN", "")

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
    allow_origins=[FRONTEND_ORIGIN, "http://127.0.0.1:3000", "https://geno-sence-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root_health() -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "genosence-backend"})


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
    session_id: str | None = None


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
        samesite="none" if IS_PRODUCTION else "lax",
        secure=IS_PRODUCTION,
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
        samesite="none" if IS_PRODUCTION else "lax",
        secure=IS_PRODUCTION,
        max_age=JWT_EXPIRE_MINUTES * 60,
    )
    return response


@app.post("/auth/signout")
def auth_signout() -> JSONResponse:
    response = JSONResponse({"ok": True})
    response.delete_cookie(
        "access_token",
        samesite="none" if IS_PRODUCTION else "lax",
        secure=IS_PRODUCTION
    )
    return response


@app.get("/auth/me")
def auth_me(request: Request) -> JSONResponse:
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return JSONResponse({"user": to_public_user(user).model_dump()})


@app.get("/user/last-upload/info")
def get_user_last_upload_info(request: Request) -> JSONResponse:
    user = get_current_user(request)
    if not user:
        return JSONResponse({"session_id": None})
    temporal_doc = _get_last_temporal_doc(user)
    if temporal_doc and "session_id" in temporal_doc:
        return JSONResponse({"session_id": temporal_doc["session_id"]})
    return JSONResponse({"session_id": None})

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
    """Upsert: one temporal-features document per user."""
    collection = get_temporal_collection()
    records = temporal_df.to_dict(orient="records")
    if not records:
        return []
    now = datetime.now(tz=timezone.utc).isoformat()
    result = collection.update_one(
        {"user_id": user_id},
        {"$set": {
            "user_id": user_id,
            "session_id": session_id,
            "records": records,
            "updated_at": now,
        }, "$setOnInsert": {"created_at": now}},
        upsert=True,
    )
    if result.upserted_id:
        return [str(result.upserted_id)]
    doc = collection.find_one({"user_id": user_id}, {"_id": 1})
    return [str(doc["_id"])] if doc else []


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
    """Upsert: one shapefile document per user."""
    collection = get_shapefiles_collection()
    now = datetime.now(tz=timezone.utc).isoformat()
    result = collection.update_one(
        {"user_id": user_id},
        {"$set": {
            "user_id": user_id,
            "session_id": session_id,
            "filename": filename,
            "geojson": geojson_data,
            "updated_at": now,
        }, "$setOnInsert": {"created_at": now}},
        upsert=True,
    )
    if result.upserted_id:
        return str(result.upserted_id)
    doc = collection.find_one({"user_id": user_id}, {"_id": 1})
    return str(doc["_id"]) if doc else ""


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


def _get_last_shapefile_doc(user: dict[str, Any]) -> dict[str, Any] | None:
    last_files = user.get("last_processed_files") or {}
    shapefile_id = last_files.get("shapefile_id")
    collection = get_shapefiles_collection()
    if shapefile_id:
        doc = collection.find_one({"_id": ObjectId(shapefile_id)})
        if doc:
            return doc
    return collection.find_one({"user_id": str(user["_id"])}, sort=[("created_at", -1)])


def _get_last_temporal_doc(user: dict[str, Any]) -> dict[str, Any] | None:
    last_files = user.get("last_processed_files") or {}
    temporal_ids = last_files.get("temporal_feature_ids") or []
    collection = get_temporal_collection()
    if temporal_ids:
        doc = collection.find_one({"_id": ObjectId(temporal_ids[0])})
        if doc:
            return doc
    return collection.find_one({"user_id": str(user["_id"])}, sort=[("created_at", -1)])


def process_reflectance_pair(rgb_path: Path, nir_path: Path, band_mapping: dict,
                              shapefile_geojson: dict, timestamp_label: str,
                              out_dir: Path) -> Path:
    """
    Placeholder: In production this would extract per-pixel band values using
    rasterio + shapely from the uploaded reflectance maps and shapefile.
    For now we produce a stub STATS CSV that the analysis engine can consume.
    Returns path to the STATS CSV.
    """
    # Read rasters and compute per-plot VI stats from pixel values.
    try:
        import rasterio
        from rasterio.mask import mask as rio_mask
        from rasterio.warp import transform_bounds
        from shapely.geometry import shape as geom_shape
        from shapely.ops import transform as transform_geom
        import pyproj
    except Exception:
        # If rasterio/shapely isn't installed, fall back to stub behavior (NaN stats)
        rows = []
        for feat in shapefile_geojson.get("features", []):
            props = feat.get("properties", {})
            plot_id = str(props.get("PLOT_ID", props.get("plot_id", "")))
            parts = plot_id.split("_")
            genotype = parts[-1] if parts else "0"
            row = {"PLOT_ID": plot_id, "genotype": genotype}
            for vi in VI_COLS:
                for stat in STAT_AGG:
                    row[f"{vi}_{stat}"] = np.nan
            rows.append(row)
        df = pd.DataFrame(rows)
        out_path = out_dir / f"{timestamp_label}_STATS.csv"
        df.to_csv(out_path, index=False)
        return out_path

    # Helper for safe statistic calculation
    def _safe_stats(arr: np.ndarray) -> dict:
        a = arr.astype(float)
        a = a[~np.isnan(a)]
        if a.size == 0:
            return {s: np.nan for s in STAT_AGG}
        return {
            "sum": float(np.nansum(a)),
            "count": int(a.size),
            "mean": float(np.nanmean(a)),
            "median": float(np.nanmedian(a)),
            "std": float(np.nanstd(a, ddof=0)),
            "var": float(np.nanvar(a, ddof=0)),
            "min": float(np.nanmin(a)),
            "max": float(np.nanmax(a)),
        }

    # Open rasters
    src_rgb = rasterio.open(rgb_path)
    src_nir = rasterio.open(nir_path)
    print(f"Opened rasters for {timestamp_label}:")
    print(f"  RGB: shape={src_rgb.shape}, count={src_rgb.count} bands, crs={src_rgb.crs}, dtype={src_rgb.dtypes}")
    print(f"  NIR: shape={src_nir.shape}, count={src_nir.count} bands, crs={src_nir.crs}, dtype={src_nir.dtypes}")
    print(f"Band mapping: {band_mapping}")

    # Get raster CRS for geometry reprojection
    raster_crs = src_rgb.crs
    if not raster_crs:
        print(f"WARNING: Raster has no CRS; assuming EPSG:4326")
        raster_crs = pyproj.CRS("EPSG:4326")
    else:
        raster_crs = pyproj.CRS.from_string(str(raster_crs))

    # Build transformer for GeoJSON (assumed EPSG:4326) -> raster CRS
    try:
        geojson_crs = pyproj.CRS("EPSG:4326")
        if geojson_crs != raster_crs:
            transformer = pyproj.Transformer.from_crs(geojson_crs, raster_crs, always_xy=True)
            print(f"Will reproject geometries from EPSG:4326 to {raster_crs}")
        else:
            transformer = None
            print(f"GeoJSON and raster CRS both {raster_crs}; no reprojection needed")
    except Exception as e:
        print(f"Failed to build transformer: {e}; no reprojection")
        transformer = None

    def reproject_geom(geom_dict, transformer):
        """Reproject GeoJSON geometry dict to raster CRS."""
        if not transformer:
            return geom_dict
        try:
            geom_shape_obj = geom_shape(geom_dict)
            reprojected = transform_geom(transformer.transform, geom_shape_obj)
            return reprojected.__geo_interface__
        except Exception as e:
            print(f"Reprojection failed: {e}")
            return geom_dict

    rows = []
    for feat in shapefile_geojson.get("features", []):
        props = feat.get("properties", {})
        plot_id = str(props.get("PLOT_ID", props.get("plot_id", "")))
        parts = plot_id.split("_")
        genotype = parts[-1] if parts else "0"
        geom = feat.get("geometry")
        if geom is None:
            print(f"{plot_id}: no geometry!")
            continue

        # Reproject geometry if needed
        geom = reproject_geom(geom, transformer)
        print(f"Processing {plot_id} with geometry type {geom.get('type')}")

        row = {"PLOT_ID": plot_id, "genotype": genotype}

        try:
            # rio_mask expects geometry as GeoJSON-like dict or shapely geometry
            # Convert dict to shapely if needed
            if isinstance(geom, dict):
                geom_for_mask = [geom_shape(geom)]
            else:
                geom_for_mask = [geom]
            
            rgb_masked, rgb_transform = rio_mask(src_rgb, geom_for_mask, crop=True, nodata=np.nan)
            print(f"  {plot_id}: RGB masked shape={rgb_masked.shape}, dtype={rgb_masked.dtype}, has NaN: {np.isnan(rgb_masked).all()}")
        except Exception as e:
            print(f"  {plot_id}: RGB masking failed: {type(e).__name__}: {e}")
            rgb_masked = None

        try:
            if isinstance(geom, dict):
                geom_for_mask = [geom_shape(geom)]
            else:
                geom_for_mask = [geom]
            
            nir_masked, nir_transform = rio_mask(src_nir, geom_for_mask, crop=True, nodata=np.nan)
            print(f"  {plot_id}: NIR masked shape={nir_masked.shape}, dtype={nir_masked.dtype}, has NaN: {np.isnan(nir_masked).all()}")
        except Exception as e:
            print(f"  {plot_id}: NIR masking failed: {type(e).__name__}: {e}")
            nir_masked = None

        # Build a dict of available bands per pixel
        bands: dict[str, np.ndarray] = {}
        # band_mapping values are 1-based indices; ensure within range
        try:
            if rgb_masked is not None:
                # rgb_masked shape = (bands, H, W)
                br = band_mapping.get("band_red", 1)
                bg = band_mapping.get("band_green", 2)
                bb = band_mapping.get("band_blue", 3)
                # Ensure indices within available bands
                for name, idx in [("Red", br), ("Green", bg), ("Blue", bb)]:
                    if 1 <= idx <= rgb_masked.shape[0]:
                        bands[name] = rgb_masked[idx - 1].astype(float)
                print(f"  {plot_id}: extracted RGB bands: {list(bands.keys())}")
        except Exception as e:
            print(f"  {plot_id}: RGB band extraction failed: {e}")

        try:
            if nir_masked is not None:
                bn = band_mapping.get("band_nir", 1)
                if 1 <= bn <= nir_masked.shape[0]:
                    bands["NIR"] = nir_masked[bn - 1].astype(float)
                    print(f"  {plot_id}: extracted NIR band, bands now: {list(bands.keys())}")
        except Exception as e:
            print(f"  {plot_id}: NIR band extraction failed: {e}")

        # Try to get Rededge from either NIR file or RGB if provided
        try:
            bre = band_mapping.get("band_rededge")
            if bre is not None:
                # check NIR first
                if nir_masked is not None and 1 <= bre <= nir_masked.shape[0]:
                    bands["Rededge"] = nir_masked[bre - 1].astype(float)
                    print(f"  {plot_id}: extracted Rededge from NIR")
                elif rgb_masked is not None and 1 <= bre <= rgb_masked.shape[0]:
                    bands["Rededge"] = rgb_masked[bre - 1].astype(float)
                    print(f"  {plot_id}: extracted Rededge from RGB")
        except Exception as e:
            print(f"  {plot_id}: Rededge extraction failed: {e}")

        print(f"  {plot_id}: final bands dict: {list(bands.keys())}")

        # Compute VIs; where input bands missing, result will be NaN
        vi_arrays: dict[str, np.ndarray] = {}
        # Create a mask of valid pixels where any band is finite
        valid_mask = None
        for b in bands.values():
            finite = np.isfinite(b)
            valid_mask = finite if valid_mask is None else (valid_mask & finite)

        # If no full-band valid_mask, allow per-formula masked calculations (use nan-safe ops)
        for vi_name, fn in VI_FORMULAS.items():
            try:
                # Prepare a small dict of arrays for lambda evaluation
                env = {}
                for key in ("NIR", "Rededge", "Red", "Blue", "Green"):
                    env[key] = bands.get(key, np.full_like(next(iter(bands.values())), np.nan)) if bands else np.array([np.nan])
                arr = fn(env)
                # Ensure array shape flattened to 1D for stats
                if isinstance(arr, np.ndarray):
                    vi_arrays[vi_name] = arr
                else:
                    vi_arrays[vi_name] = np.array(arr)
                # Check if any finite values
                if isinstance(arr, np.ndarray):
                    finite_count = np.isfinite(arr).sum()
                    print(f"  {plot_id}: {vi_name} has {finite_count} finite values out of {arr.size}")
            except Exception as e:
                print(f"  {plot_id}: VI formula {vi_name} failed: {e}")
                vi_arrays[vi_name] = np.array([np.nan])

        # Calculate stats for each VI and add to row
        for vi, arr in vi_arrays.items():
            # Flatten and drop NaNs
            if arr is None:
                stats = {s: np.nan for s in STAT_AGG}
            else:
                flat = arr.flatten()
                stats = _safe_stats(flat)
            for s, val in stats.items():
                row[f"{vi}_{s}"] = val

        rows.append(row)

    # Close datasets
    try:
        src_rgb.close()
    except Exception:
        pass
    try:
        src_nir.close()
    except Exception:
        pass

    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"/upload/reflectance-maps: extracted values for {timestamp_label}")
        print(df.to_string(index=False))
    else:
        print(f"/upload/reflectance-maps: no plot rows extracted for {timestamp_label}")
    out_path = out_dir / f"{timestamp_label}_STATS.csv"
    df.to_csv(out_path, index=False)
    return out_path


def build_temporal_summary_from_stats(
    stats_paths: list[Path],
    geojson_data: dict[str, Any],
) -> pd.DataFrame:
    """Aggregate per-timestamp STATS CSVs into one per-plot temporal table."""
    base_rows: list[dict[str, Any]] = []
    for feat in geojson_data.get("features", []):
        props = feat.get("properties", {})
        plot_id = str(props.get("PLOT_ID", props.get("plot_id", "")))
        if not plot_id:
            continue
        parts = plot_id.split("_")
        genotype = parts[-1] if parts else "0"
        base_rows.append({
            "PLOT_ID": plot_id,
            "genotype": genotype,
            "Yield": props.get("Yield", np.nan),
        })

    base_df = pd.DataFrame(base_rows).drop_duplicates("PLOT_ID")
    if base_df.empty:
        return base_df

    long_frames: list[pd.DataFrame] = []
    for ts_idx, stats_path in enumerate(stats_paths, start=1):
        try:
            ts_df = pd.read_csv(stats_path)
        except Exception:
            continue
        ts_df.columns = ts_df.columns.str.strip()
        if "PLOT_ID" not in ts_df.columns:
            continue
        ts_df = ts_df.copy()
        ts_df["timestamp_idx"] = ts_idx
        long_frames.append(ts_df)

    if not long_frames:
        return base_df

    long_df = pd.concat(long_frames, ignore_index=True, sort=False)
    long_df["PLOT_ID"] = long_df["PLOT_ID"].astype(str)

    summary = base_df.copy()
    for vi in VI_COLS:
        mean_col = f"{vi}_mean"
        if mean_col not in long_df.columns:
            continue
        working = long_df[["PLOT_ID", mean_col]].copy()
        working[mean_col] = pd.to_numeric(working[mean_col], errors="coerce")
        agg = working.groupby("PLOT_ID")[mean_col].agg(["mean", "std", "max", "min"]).reset_index()
        agg = agg.rename(columns={
            "mean": f"{vi}_mean",
            "std": f"{vi}_std",
            "max": f"{vi}_max",
            "min": f"{vi}_min",
        })
        summary = summary.merge(agg, on="PLOT_ID", how="left")

    return summary


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

        # Keep source tif files local only. We upload only extracted CSV values.
        if user_id:
            print(
                f"/upload/reflectance-maps: keeping source rasters local for {label}; "
                f"not uploading {rgb_file.filename} / {nir_file.filename} to cloud storage"
            )

        stats_df = process_reflectance_pair(
            rgb_path, nir_path, band_info, geojson_data, label, ts_dir
        )
        stats_files.append(stats_df.name if isinstance(stats_df, Path) else str(stats_df))

        try:
            extracted_df = pd.read_csv(ts_dir / stats_files[-1])
            print(f"/upload/reflectance-maps: timestamp {label} extracted CSV rows")
            print(extracted_df.to_string(index=False))
        except Exception as exc:
            print(f"/upload/reflectance-maps: failed to read extracted stats for {label}: {exc}")

    # Build a temporal CSV from per-timestamp VI statistics.
    temporal_df = build_temporal_summary_from_stats([ts_dir / name for name in stats_files], geojson_data)
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
    warnings_list: list[str] = []

    for i, ts_file in enumerate(timestamp_csvs):
        label = _extract_date_label(ts_file.filename or f"t{i+1}")
        if not label:
            label = f"t{i+1}"
        try:
            ts_bytes = await ts_file.read()
            df_ts = pd.read_csv(io.BytesIO(ts_bytes))
            df_ts.columns = df_ts.columns.str.strip()

            # Detect format: does this CSV already have pre-computed _mean stats?
            has_precomputed = any(c.endswith("_mean") for c in df_ts.columns)
            has_plot_id = "PLOT_ID" in df_ts.columns

            if has_precomputed and has_plot_id:
                # Format B: already-computed VI stats (e.g. sample timestamp files)
                # Write directly as STATS CSV
                out_name = f"final_{label}_STATS.csv"
                df_ts.to_csv(ts_dir / out_name, index=False)
                processed_labels.append(label)
                warnings_list.append(
                    f"Timestamp {i+1} ({ts_file.filename}): used pre-computed VI stats directly."
                )
            else:
                # Format A: raw pixel CSV — try to compute VIs
                required_band_cols = {"Red", "Green", "Blue", "NIR", "Rededge"}
                missing = required_band_cols - set(df_ts.columns)
                if missing:
                    errors.append(
                        f"Timestamp {i+1} ({ts_file.filename}): CSV missing columns: {missing}. "
                        "Upload either raw pixel CSVs (Red/Green/Blue/NIR/Rededge) or pre-computed STATS CSVs (_mean columns)."
                    )
                    continue
                pixel_path = tmp_dir / f"pixels_{i}.csv"
                pixel_path.write_bytes(ts_bytes)
                from analysis_helpers import compute_vi_stats  # type: ignore[import]
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
        "warnings": warnings_list,
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
# Upload: Mode D – Shapefile only (visualize on map)
# ─────────────────────────────────────────────────────────────────

@app.post("/upload/shapefile-only")
async def upload_shapefile_only(
    request: Request,
    shapefile_json: UploadFile = File(..., description="Shapefile as GeoJSON"),
) -> JSONResponse:
    """
    Accept a GeoJSON shapefile and store it for map visualization.
    Creates a minimal temporal CSV (PLOT_ID + genotype) from the shapefile
    properties so the session can be used for other endpoints without crashing.
    """
    import json as json_module

    session_id = str(uuid.uuid4())
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"session_{session_id}_"))
    ts_dir = tmp_dir / "timestamps"
    ts_dir.mkdir()

    geojson_bytes = await shapefile_json.read()
    try:
        geojson_data = json_module.loads(geojson_bytes.decode("utf-8"))
    except Exception as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(400, f"Invalid GeoJSON: {exc}")

    feature_count = len(geojson_data.get("features", []))
    geojson_path = tmp_dir / "plots.geojson"
    geojson_path.write_text(json_module.dumps(geojson_data))

    # Build minimal temporal CSV from shapefile feature properties
    rows = []
    for feat in geojson_data.get("features", []):
        props = feat.get("properties", {})
        plot_id = str(props.get("PLOT_ID", props.get("plot_id", props.get("Plot_ID", ""))))
        raw_geno = props.get("genotype", props.get("GENOTYPE", ""))
        if raw_geno not in (None, ""):
            genotype = raw_geno
        elif "_" in plot_id:
            genotype = plot_id.split("_")[-1]
        else:
            genotype = 0
        rows.append({"PLOT_ID": plot_id, "genotype": genotype})

    temporal_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["PLOT_ID", "genotype"])
    temporal_path = tmp_dir / "temporalDataSet.csv"
    temporal_df.to_csv(temporal_path, index=False)

    _sessions[session_id] = {
        "temporal_csv": temporal_path,
        "timestamps_dir": ts_dir,
        "geojson": geojson_path,
        "mode": "shapefile-only",
        "timestamp_count": 0,
        "timestamp_labels": [],
    }
    analysis.invalidate_session_cache(session_id)

    user = get_current_user(request)
    shapefile_id = None
    if user:
        user_id = str(user["_id"])
        ensure_user_r2_prefix(user_id)
        shapefile_key = None
        temporal_ids: list[str] = []
        try:
            shapefile_key = upload_bytes_to_r2(user_id, geojson_bytes, shapefile_json.filename)
        except Exception:
            pass
        try:
            delete_user_upload_data(user_id)
            temporal_ids = store_temporal_features(user_id, session_id, temporal_df)
            shapefile_id = store_shapefile_geojson(user_id, session_id, geojson_data, shapefile_json.filename)
        except Exception as exc:
            print(f"/upload/shapefile-only: Mongo write failed: {exc}")
        update_user_last_processed_files(
            user_id, None, shapefile_key, [], [],
            temporal_ids=temporal_ids, shapefile_id=shapefile_id,
        )

    return JSONResponse({
        "session_id": session_id,
        "feature_count": feature_count,
        "has_shapefile": True,
        "message": f"Shapefile loaded with {feature_count} features. Redirect to dashboard to visualize.",
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


@app.get("/user/last-upload/geojson", response_model=None)
def user_last_upload_geojson(request: Request) -> JSONResponse:
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    doc = _get_last_shapefile_doc(user)
    if not doc:
        raise HTTPException(status_code=404, detail="No shapefile uploaded")
    geojson_data = doc.get("geojson")
    if not geojson_data:
        raise HTTPException(status_code=404, detail="No shapefile data")
    return JSONResponse(geojson_data)


@app.get("/user/last-upload/temporal-csv")
def user_last_upload_temporal_csv(request: Request) -> StreamingResponse:
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    doc = _get_last_temporal_doc(user)
    if not doc:
        raise HTTPException(status_code=404, detail="No temporal dataset uploaded")
    records = doc.get("records") or []
    if not records:
        raise HTTPException(status_code=404, detail="No temporal dataset records")
    frame = pd.DataFrame(records)
    csv_text = frame.to_csv(index=False)
    buffer = io.BytesIO(csv_text.encode("utf-8"))
    return StreamingResponse(buffer, media_type="text/csv")


@app.get("/user/last-upload/info")
def user_last_upload_info(request: Request) -> JSONResponse:
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    doc = _get_last_temporal_doc(user)
    if not doc:
        raise HTTPException(status_code=404, detail="No upload found")
    return JSONResponse({
        "session_id": doc.get("session_id"),
    })


@app.post("/user/reset-upload")
def reset_user_upload(request: Request) -> JSONResponse:
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        delete_user_upload_data(str(user["_id"]))
        update_user_last_processed_files(str(user["_id"]), None, None, [], [])
        return JSONResponse({"message": "User upload data reset successfully. Dashboard will now show sample data."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────
# Helper: resolve analysis engine for session or default
# ─────────────────────────────────────────────────────────────────

def _resolve_engine(session_id: str | None) -> analysis.AnalysisEngine:
    import json as _json, tempfile as _tmp
    # 1. Session is alive in memory
    if session_id and session_id in _sessions:
        sess = _sessions[session_id]
        return analysis.AnalysisEngine(
            temporal_csv=sess["temporal_csv"],
            timestamps_dir=sess["timestamps_dir"],
            session_id=session_id,
        )
    # 2. Session ID provided but server restarted — try to rebuild from MongoDB
    if session_id and mongo_client:
        try:
            temporal_doc = get_temporal_collection().find_one({"session_id": session_id})
            shapefile_doc = get_shapefiles_collection().find_one({"session_id": session_id})
            if temporal_doc:
                tmp_dir = Path(_tmp.mkdtemp(prefix=f"restore_{session_id}_"))
                ts_dir = tmp_dir / "timestamps"
                ts_dir.mkdir()
                records = temporal_doc.get("records", [])
                temporal_df = pd.DataFrame(records) if records else pd.DataFrame()
                temporal_path = tmp_dir / "temporalDataSet.csv"
                temporal_df.to_csv(temporal_path, index=False)
                ts_labels: list[str] = []
                ts_doc = get_timestamps_collection().find_one({
                    "session_id": session_id,
                    "user_id": temporal_doc.get("user_id"),
                })
                if ts_doc:
                    import re as _re
                    for idx, ts in enumerate(ts_doc.get("timestamps", []), start=1):
                        label = str(ts.get("date") or f"t{idx}")
                        safe_label = _re.sub(r"[^0-9A-Za-z_\-]", "_", label)
                        ts_labels.append(safe_label)
                        ts_records = ts.get("records", [])
                        if ts_records:
                            ts_df = pd.DataFrame(ts_records)
                            ts_df.to_csv(ts_dir / f"final_{safe_label}.csv", index=False)
                geojson_path: Path | None = None
                if shapefile_doc:
                    geojson_path = tmp_dir / "plots.geojson"
                    geojson_path.write_text(_json.dumps(shapefile_doc.get("geojson", {})))
                _sessions[session_id] = {
                    "temporal_csv": temporal_path,
                    "timestamps_dir": ts_dir,
                    "geojson": geojson_path,
                    "mode": "restored",
                    "timestamp_count": len(ts_labels),
                    "timestamp_labels": ts_labels,
                }
                analysis.invalidate_session_cache(session_id)
                print(f"_resolve_engine: restored session {session_id} from MongoDB")
                return analysis.AnalysisEngine(
                    temporal_csv=temporal_path,
                    timestamps_dir=ts_dir,
                    session_id=session_id,
                )
        except Exception as exc:
            print(f"_resolve_engine: MongoDB restore failed for {session_id}: {exc}")
    # 3. No session — fall back to user's last upload, then sample data
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


@app.get("/temporal/time-series")
def temporal_time_series(session_id: str | None = Query(default=None)) -> JSONResponse:
    return JSONResponse(_resolve_engine(session_id).get_time_series())


@app.get("/temporal/ols-slope")
def temporal_ols_slope(session_id: str | None = Query(default=None)) -> JSONResponse:
    return JSONResponse(_resolve_engine(session_id).get_ols_slope_effect())


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

# ── Vegetation-index reference (injected into every chat system prompt) ─────────
VI_GLOSSARY = """
VEGETATION INDEX REFERENCE (authoritative definitions — always use these):
• NDVI  (Normalized Difference Vegetation Index): measures green biomass / photosynthetic
  activity.  Range −1 to +1.  Higher = healthier, denser canopy.  Drops sharply during
  senescence or water stress.
• NDWI  (Normalized Difference Water Index): measures LIQUID WATER CONTENT in the canopy.
  Higher NDWI = more water stored in leaves = lower water stress.  Negative or very low
  NDWI = drought / dehydration stress.
• NDRE  (Normalized Difference Red Edge): sensitive to chlorophyll concentration, especially
  useful for detecting early nutrient stress before NDVI changes.  Higher = more chlorophyll.
• MTCI  (MERIS Terrestrial Chlorophyll Index): tracks chlorophyll via red-edge contrast.
  High MTCI ↔ abundant chlorophyll and active photosynthesis.
• NDCI  (Normalized Difference Chlorophyll Index): alternative chlorophyll proxy (red-edge/red
  ratio).  Higher = more chlorophyll.
• MSI   (Moisture Stress Index): Red/NIR ratio — INVERSE of water status.  HIGH MSI = high
  water stress / moisture deficit.  LOW MSI = well-watered crop.
• WI    (Water Index): NIR/Green ratio — higher = more water content in vegetation.
• SAVI  (Soil-Adjusted Vegetation Index): NDVI corrected for soil background reflectance;
  more reliable at low canopy cover.
• EVI   (Enhanced Vegetation Index): improved NDVI for high-biomass / high-LAI conditions;
  less prone to saturation.
• EXG   (Excess Green Index): 2G−R−B — emphasises green pigmentation; used for canopy
  cover estimation in RGB imagery.
• CIgreen (Chlorophyll Index Green): (NIR/Green)−1; tracks chlorophyll content from green
  reflectance.
• PSRI  (Plant Senescence Reflectance Index): (Red−Green)/NIR — rises as carotenoids
  increase relative to chlorophyll; HIGH PSRI = active senescence / leaf aging.
• SIPI  (Structure Insensitive Pigment Index): carotenoid-to-chlorophyll ratio proxy;
  HIGH SIPI = stressed or senescing tissue.
• Yield_Class: categorical performance bin — Low / Medium / High — derived from the
  33rd and 66th percentile of mean yield across all genotypes.
• CV (Coefficient of Variation): yield variability across plots.  <10% = very stable;
  10–25% = moderate; >25% = high variation.
"""


def _build_csv_context(session_id: str | None, extra_context: str | None) -> str:
    """Return the temporal CSV as a text table for the system prompt (max 100 rows)."""
    csv_block = ""

    def _format_df(df: pd.DataFrame) -> str:
        df.columns = df.columns.str.strip()
        if len(df) > 100:
            df = df.sample(n=100, random_state=42)
        return df.to_csv(index=False)

    # 1. Try session in memory
    if session_id and session_id in _sessions:
        csv_path = _sessions[session_id].get("temporal_csv")
        if csv_path and Path(csv_path).exists():
            try:
                df = pd.read_csv(csv_path)
                csv_block = _format_df(df)
            except Exception:
                pass

    # 2. Try MongoDB if session not in memory
    if not csv_block and session_id and mongo_client:
        try:
            doc = get_temporal_collection().find_one({"session_id": session_id})
            if doc and doc.get("records"):
                df = pd.DataFrame(doc["records"])
                csv_block = _format_df(df)
        except Exception:
            pass

    # 3. Fallback → load the FULL sample temporal CSV from disk
    if not csv_block:
        try:
            sample_csv = SAMPLES_DIR / "temporalDataSet.csv"
            df_sample = pd.read_csv(sample_csv)
            csv_block = _format_df(df_sample)
        except Exception:
            pass
    # Last resort: compact engine summary
    if not csv_block:
        try:
            eng = _resolve_engine(session_id)
            ctx = eng.get_chat_context()
            csv_block = ctx.get("context", "")
        except Exception:
            pass

    if extra_context:
        csv_block = (csv_block + "\n" + extra_context).strip()

    return csv_block


@app.post("/chat")
def chat(req: ChatRequest) -> JSONResponse:
    if not openai_client:
        raise HTTPException(status_code=500, detail="GPT_API_KEY is not set")

    # Build data context from the session CSV (or sample data if no session)
    data_context = _build_csv_context(req.session_id, req.context)

    system_prompt = (
        "You are GenoSense AI — an agricultural crop-science assistant embedded in a "
        "precision-phenotyping dashboard. You have access to the full experiment CSV below.\n"
        "\n"
        "STRICT RESPONSE RULES — follow every rule exactly, no exceptions:\n"
        "RULE 1 — GENOTYPE QUERIES: If the user asks which genotype is best/worst/highest/lowest "
        "for ANY trait, look up the data, find the answer, and reply with ONLY:\n"
        "  'Genotype <number>' (e.g. 'Genotype 47')\n"
        "  You MAY add one sentence of supporting data (e.g. the exact index value). "
        "  Do NOT ask for clarification. Do NOT say you cannot determine it. "
        "  Do NOT suggest uploading more data. Just pick the best one from the CSV.\n"
        "RULE 2 — VALUE QUERIES: If asked for a specific value (yield, NDVI, etc.), "
        "return ONLY the number from the CSV. Example: '0.742'\n"
        "RULE 3 — EXPLANATION QUERIES: If asked what an index MEANS (not a data lookup), "
        "give a 1-2 sentence definition using the VI Reference below.\n"
        "RULE 4 — NEVER say 'I cannot determine', 'insufficient data', or 'please upload'. "
        "The CSV is always provided — use it.\n"
        "RULE 5 — Keep all answers under 3 sentences unless a table/breakdown is explicitly requested.\n"
        f"\n{VI_GLOSSARY}"
    )

    if data_context:
        system_prompt += (
            "\n\n━━ EXPERIMENT DATA (full temporal CSV) ━━\n"
            "This is the complete dataset. Use it to answer ALL data questions.\n"
            + data_context
        )

    history = clamp_history(req.history)
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    messages += [{"role": m.role, "content": m.content} for m in history]
    messages.append({"role": "user", "content": req.message})

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,   # zero temperature = maximally factual / deterministic
        max_tokens=700,
    )
    reply = response.choices[0].message.content or ""
    return JSONResponse({"reply": reply.strip()})

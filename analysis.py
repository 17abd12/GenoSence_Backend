"""
Runtime Temporal Analysis Engine
Reads temporalDataSet.csv + 16 timestamp CSVs and computes everything on-demand.
No stored intermediate CSVs — all calculations happen at request time.
Fully feature-agnostic: works with any VI columns present in the CSVs.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, pearsonr, spearmanr

warnings.filterwarnings("ignore")

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
TEMPORAL_CSV = SAMPLES_DIR / "temporalDataSet.csv"
TIMESTAMPS_DIR = SAMPLES_DIR / "timestamps"

TIME_DATES_STR = [
    "2023-11-17", "2023-11-24", "2023-12-01", "2023-12-21",
    "2023-12-29", "2024-01-05", "2024-01-12", "2024-01-19",
    "2024-01-26", "2024-02-02", "2024-02-16", "2024-02-23",
    "2024-02-29", "2024-03-29", "2024-04-04", "2024-04-22",
]

# ─────────────────────────────────────────────────────────────────
# IN-PROCESS CACHE (reset on each server restart)
# ─────────────────────────────────────────────────────────────────
_cache: dict[str, Any] = {}


def _load_temporal() -> pd.DataFrame:
    if "temporal" not in _cache:
        df = pd.read_csv(TEMPORAL_CSV)
        df.columns = df.columns.str.strip()
        _cache["temporal"] = df
    return _cache["temporal"].copy()


def _load_timestamps_wide() -> pd.DataFrame:
    """Merge 16 timestamp CSVs into one wide df with PLOT_ID + VI_t1..t16 columns."""
    if "ts_wide" not in _cache:
        files = sorted(TIMESTAMPS_DIR.glob("final_*.csv"))
        parts: list[pd.DataFrame] = []
        for idx, f in enumerate(files, start=1):
            label = f"t{idx}"
            df = pd.read_csv(f)
            df.columns = df.columns.str.strip()
            mean_cols = [c for c in df.columns if c.endswith("_mean")]
            rename: dict[str, str] = {}
            for c in mean_cols:
                base = c[:-5]
                rename[c] = f"{base}_{label}"
            for extra in ("stage", "experiment", "genotype", "fractional_cover",
                          "average_height_m", "LPT_70", "ALS", "WALS"):
                if extra in df.columns and extra not in rename:
                    rename[extra] = f"{extra}_{label}"
            df = df.rename(columns=rename)
            keep = ["PLOT_ID"] + [v for v in rename.values() if v in df.columns]
            parts.append(df[[c for c in keep if c in df.columns]])

        wide = parts[0]
        for part in parts[1:]:
            dup = [c for c in part.columns if c != "PLOT_ID" and c in wide.columns]
            part = part.drop(columns=dup, errors="ignore")
            wide = wide.merge(part, on="PLOT_ID", how="outer")
        _cache["ts_wide"] = wide
    return _cache["ts_wide"].copy()


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def _detect_vi_bases(df: pd.DataFrame) -> list[str]:
    """Auto-detect VI names from *_mean columns."""
    bases = sorted({c[:-5] for c in df.columns if c.endswith("_mean")})
    exclude = {"id", "PLOT_ID", "genotype", "experiment"}
    return [b for b in bases if b not in exclude]


def _safe_float(v: Any) -> Any:
    if v is None:
        return None
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 6)
    except Exception:
        return None


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    clean = df.copy()
    for col in clean.select_dtypes(include=[np.number]).columns:
        clean[col] = clean[col].apply(_safe_float)
    clean = clean.where(clean.notna(), other=None)
    return clean.to_dict(orient="records")


def _yield_class(y: float, t33: float, t66: float) -> str:
    if y <= t33:
        return "Low"
    if y <= t66:
        return "Medium"
    return "High"


def _prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """Filter + compute stable genotype means + Yield_Class. Returns (filtered, geno_mean, t33, t66)."""
    counts = df["genotype"].value_counts()
    eligible = counts[counts >= 3].index
    filtered = df[df["genotype"].isin(eligible)].copy()

    stab = (
        filtered.groupby("genotype")
        .agg(my=("Yield", "mean"), sy=("Yield", "std"))
        .assign(cv=lambda x: (x["sy"] / x["my"].abs()) * 100)
    )
    stable_genos = stab[stab["cv"] < 25].index
    stable = filtered[filtered["genotype"].isin(stable_genos)].copy()
    if stable.empty:
        stable = filtered.copy()

    num_cols = stable.select_dtypes(include=np.number).columns.tolist()
    gm = stable.groupby("genotype")[num_cols].mean().reset_index()

    t33 = float(np.percentile(gm["Yield"].dropna(), 33))
    t66 = float(np.percentile(gm["Yield"].dropna(), 66))
    gm["Yield_Class"] = gm["Yield"].apply(lambda y: _yield_class(y, t33, t66))
    return filtered, gm, t33, t66


# ─────────────────────────────────────────────────────────────────
# ENDPOINT IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────────

def get_stability() -> dict:
    """4A – Stability classification (CV-based bar chart + table)."""
    df = _load_temporal()
    counts = df["genotype"].value_counts()
    eligible = counts[counts >= 3].index
    filtered = df[df["genotype"].isin(eligible)].copy()

    summary = (
        filtered.groupby("genotype")
        .agg(mean_yield=("Yield", "mean"), sd_yield=("Yield", "std"), plot_count=("PLOT_ID", "count"))
        .reset_index()
    )
    summary["cv"] = (summary["sd_yield"] / summary["mean_yield"].abs()) * 100

    def cv_cat(cv: float) -> str:
        if cv < 10:
            return "Low variation (<10%)"
        if cv <= 25:
            return "Moderate variation (10–25%)"
        return "High variation (>25%)"

    summary["cv_category"] = summary["cv"].apply(cv_cat)
    cat_counts = summary["cv_category"].value_counts().reset_index()
    cat_counts.columns = ["cv_category", "count"]
    return {
        "summary": _df_to_records(summary[["genotype", "mean_yield", "cv", "cv_category", "plot_count"]]),
        "category_counts": _df_to_records(cat_counts),
    }


def get_yield_class() -> dict:
    """4B – Yield class distribution (bar + table)."""
    df = _load_temporal()
    _, gm, t33, t66 = _prepare(df)
    out = gm[["genotype", "Yield", "Yield_Class"]].sort_values("Yield", ascending=False)
    counts = out["Yield_Class"].value_counts().reset_index()
    counts.columns = ["Yield_Class", "count"]
    order = {"Low": 0, "Medium": 1, "High": 2}
    counts["_o"] = counts["Yield_Class"].map(order)
    counts = counts.sort_values("_o").drop(columns=["_o"])
    return {
        "thresholds": {"t33": _safe_float(t33), "t66": _safe_float(t66)},
        "distribution": _df_to_records(counts),
        "genotype_table": _df_to_records(out),
    }


def get_tukey() -> dict:
    """4D – Significant features by ANOVA p-value (Tukey-style summary)."""
    df = _load_temporal()
    _, gm, _, _ = _prepare(df)
    vi_bases = _detect_vi_bases(df)
    rows = []
    for base in vi_bases:
        col = f"{base}_mean"
        if col not in gm.columns:
            continue
        grps = {cls: g[col].dropna().values for cls, g in gm.groupby("Yield_Class")}
        if not all(len(g) > 1 for g in grps.values()):
            continue
        try:
            p = f_oneway(*grps.values()).pvalue
            if p < 0.05:
                cls_means = {k: _safe_float(float(np.mean(v))) for k, v in grps.items()}
                pairs = []
                ks = list(grps.keys())
                for i in range(len(ks)):
                    for j in range(i + 1, len(ks)):
                        pairs.append(f"{ks[i]}-{ks[j]}")
                rows.append({
                    "feature": base,
                    "p_value": _safe_float(p),
                    "sig_count": len(pairs),
                    "significant_pairs": ", ".join(pairs),
                    **{f"mean_{k}": v for k, v in cls_means.items()},
                })
        except Exception:
            pass
    rows.sort(key=lambda r: r["p_value"] or 1.0)
    return {"tukey": rows}


def get_correlation() -> dict:
    """4E – Pearson/Spearman correlation of each *_mean feature with Yield."""
    df = _load_temporal()
    _, gm, _, _ = _prepare(df)
    vi_bases = _detect_vi_bases(df)
    rows = []
    for base in vi_bases:
        col = f"{base}_mean"
        if col not in gm.columns:
            continue
        sub = gm[["Yield", col]].dropna()
        if len(sub) < 5:
            continue
        try:
            r_p, p_p = pearsonr(sub["Yield"], sub[col])
            r_s, p_s = spearmanr(sub["Yield"], sub[col])
            rows.append({
                "feature": base,
                "pearson_r": _safe_float(r_p),
                "pearson_p": _safe_float(p_p),
                "spearman_r": _safe_float(r_s),
                "spearman_p": _safe_float(p_s),
            })
        except Exception:
            pass
    rows.sort(key=lambda r: abs(r["pearson_r"] or 0), reverse=True)
    return {"correlations": rows[:20]}  # top 20


def get_growth_senescence() -> dict:
    """4F – Growth and senescence rates by yield class from timestamp CSVs."""
    df = _load_temporal()
    _, gm, _, _ = _prepare(df)
    ts_wide = _load_timestamps_wide()
    geno_info = df[["PLOT_ID", "genotype"]].drop_duplicates("PLOT_ID")
    merged = ts_wide.merge(geno_info, on="PLOT_ID", how="left")
    stable_genos = gm["genotype"].unique()
    merged = merged[merged["genotype"].isin(stable_genos)]

    vi_bases = _detect_vi_bases(df)
    dates = [pd.Timestamp(d) for d in TIME_DATES_STR]

    records = []
    for geno, grp in merged.groupby("genotype"):
        row_gm = gm[gm["genotype"] == geno]
        if row_gm.empty:
            continue
        row: dict[str, Any] = {
            "genotype": int(geno),
            "Yield_Class": str(row_gm["Yield_Class"].values[0]),
            "Yield": _safe_float(row_gm["Yield"].values[0]),
        }
        for feat in vi_bases:
            feat_cols = [f"{feat}_t{i}" for i in range(1, 17) if f"{feat}_t{i}" in grp.columns]
            if len(feat_cols) < 2:
                continue
            vals = grp[feat_cols].mean(axis=0).values
            n = len(feat_cols)
            day_arr = np.array([(dates[i] - dates[0]).days for i in range(n)])
            delta_v = np.diff(vals)
            delta_d = np.diff(day_arr)
            delta_d = np.where(delta_d == 0, 1, delta_d)
            rate = delta_v / delta_d
            mid = len(rate) // 2
            row[f"{feat}_growth"] = _safe_float(float(np.nanmean(rate[:mid])))
            row[f"{feat}_senescence"] = _safe_float(float(np.nanmean(rate[mid:])))
        records.append(row)

    result_df = pd.DataFrame(records)
    rate_cols = [c for c in result_df.columns if c.endswith("_growth") or c.endswith("_senescence")]
    melt = result_df[["genotype", "Yield_Class"] + rate_cols].melt(
        id_vars=["genotype", "Yield_Class"], var_name="feature_rate", value_name="rate_value"
    )
    melt["rate_type"] = melt["feature_rate"].apply(
        lambda c: "growth" if c.endswith("_growth") else "senescence"
    )
    melt["feature"] = melt["feature_rate"].apply(lambda c: c.rsplit("_", 1)[0])
    melt["rate_value"] = melt["rate_value"].apply(_safe_float)
    return {
        "rates": _df_to_records(melt[["genotype", "Yield_Class", "feature", "rate_type", "rate_value"]]),
        "features": sorted(vi_bases),
    }


def get_phenology() -> dict:
    """4G – Phenology features (peak, time_to_peak, senescence_duration, staygreen_auc)."""
    df = _load_temporal()
    _, gm, _, _ = _prepare(df)
    ts_wide = _load_timestamps_wide()
    geno_info = df[["PLOT_ID", "genotype"]].drop_duplicates("PLOT_ID")
    merged = ts_wide.merge(geno_info, on="PLOT_ID", how="left")
    stable_genos = gm["genotype"].unique()
    merged = merged[merged["genotype"].isin(stable_genos)]

    vi_bases = _detect_vi_bases(df)
    dates = [pd.Timestamp(d) for d in TIME_DATES_STR]
    days_arr = np.array([(d - dates[0]).days for d in dates])

    records = []
    for geno, grp in merged.groupby("genotype"):
        row_gm = gm[gm["genotype"] == geno]
        if row_gm.empty:
            continue
        row: dict[str, Any] = {
            "genotype": int(geno),
            "Yield_Class": str(row_gm["Yield_Class"].values[0]),
            "Yield": _safe_float(row_gm["Yield"].values[0]),
        }
        for feat in vi_bases:
            feat_cols = [f"{feat}_t{i}" for i in range(1, 17) if f"{feat}_t{i}" in grp.columns]
            if len(feat_cols) < 3:
                continue
            n = len(feat_cols)
            vals = grp[feat_cols].mean(axis=0).values
            d_sub = days_arr[:n]
            peak_idx = int(np.nanargmax(vals))
            row[f"{feat}_peak"] = _safe_float(float(vals[peak_idx]))
            row[f"{feat}_time_to_peak_days"] = int(d_sub[peak_idx])
            row[f"{feat}_senescence_duration_days"] = int(d_sub[-1] - d_sub[peak_idx])
            post = vals[peak_idx:]
            try:
                auc = float(np.trapezoid(post, x=d_sub[peak_idx:]))
            except Exception:
                auc = float(np.trapz(post, x=d_sub[peak_idx:]))
            row[f"{feat}_staygreen_auc"] = _safe_float(auc)
        records.append(row)

    result_df = pd.DataFrame(records)
    id_cols = ["genotype", "Yield_Class", "Yield"]
    metric_cols = [c for c in result_df.columns if c not in id_cols]

    def parse_metric(col: str) -> tuple[str, str]:
        for suffix in ("_staygreen_auc", "_senescence_duration_days", "_time_to_peak_days", "_peak"):
            if col.endswith(suffix):
                return col[: -len(suffix)], suffix[1:]
        return col, "unknown"

    melt = result_df[id_cols + metric_cols].melt(
        id_vars=id_cols, var_name="metric_col", value_name="value"
    )
    parsed = melt["metric_col"].apply(parse_metric)
    melt["feature"] = [p[0] for p in parsed]
    melt["metric"] = [p[1] for p in parsed]
    melt = melt.drop(columns=["metric_col"])
    melt["value"] = melt["value"].apply(_safe_float)
    return {
        "records": _df_to_records(melt),
        "features": sorted(vi_bases),
        "metrics": ["peak", "time_to_peak_days", "senescence_duration_days", "staygreen_auc"],
    }


def get_feature_interpretation() -> dict:
    """4I – Biological interpretation of significant features."""
    df = _load_temporal()
    _, gm, _, _ = _prepare(df)
    vi_bases = _detect_vi_bases(df)
    BIO: dict[str, str] = {
        "_mean": "Seasonal mean — overall crop performance indicator",
        "_slope": "Temporal trend — positive=growth, negative=decline",
        "_max_increase": "Peak growth flush during the season",
        "_max_decrease": "Fastest senescence step",
        "_peak": "Maximum photosynthetic capacity reached",
    }
    rows = []
    for base in vi_bases:
        col = f"{base}_mean"
        if col not in gm.columns:
            continue
        sub = gm[["Yield_Class", col]].dropna()
        if sub.empty:
            continue
        try:
            groups = [g[col].values for _, g in sub.groupby("Yield_Class")]
            if not all(len(g) > 1 for g in groups):
                continue
            p = f_oneway(*groups).pvalue
            if p >= 0.05:
                continue
            cls_means = sub.groupby("Yield_Class")[col].mean()
            high_m = float(cls_means.get("High", np.nan))
            low_m = float(cls_means.get("Low", np.nan))
            direction = "Positive" if (not np.isnan(high_m) and not np.isnan(low_m) and high_m > low_m) else "Negative"
            reason = next((v for k, v in BIO.items() if k in col), "Derived temporal feature")
            rows.append({
                "feature": base,
                "p_value": _safe_float(p),
                "mean_high": _safe_float(high_m),
                "mean_low": _safe_float(low_m),
                "effect_direction": direction,
                "biological_reason": reason,
            })
        except Exception:
            pass
    rows.sort(key=lambda r: r["p_value"] or 1.0)
    return {"interpretations": rows}


def get_outliers(yield_classes: list[str] | None = None) -> dict:
    """4K – Genotype outlier Z-scores."""
    df = _load_temporal()
    _, gm, _, _ = _prepare(df)
    vi_bases = _detect_vi_bases(df)
    mean_cols = [f"{b}_mean" for b in vi_bases if f"{b}_mean" in gm.columns]

    work = gm[gm["Yield_Class"].isin(yield_classes)].copy() if yield_classes else gm.copy()
    outlier_rows = []
    for yc, sub in work.groupby("Yield_Class"):
        cls_mean = sub[mean_cols].mean()
        cls_std = sub[mean_cols].std()
        for _, row in sub.iterrows():
            for col in mean_cols:
                val = row[col]
                m, s = cls_mean[col], cls_std[col]
                if s == 0 or np.isnan(val) or np.isnan(m):
                    continue
                z = (val - m) / s
                if abs(z) > 2.0:
                    outlier_rows.append({
                        "genotype": int(row["genotype"]),
                        "Yield_Class": str(yc),
                        "Feature": col[:-5],
                        "Value": _safe_float(val),
                        "Class_Mean": _safe_float(m),
                        "Z_score": _safe_float(z),
                    })

    out_df = pd.DataFrame(outlier_rows) if outlier_rows else pd.DataFrame()
    heatmap: list[dict] = []
    if not out_df.empty:
        piv = (
            out_df.groupby(["Yield_Class", "Feature"])["Z_score"]
            .apply(lambda x: float(np.mean(np.abs(x))))
            .reset_index()
            .rename(columns={"Z_score": "mean_abs_z"})
        )
        heatmap = _df_to_records(piv)

    return {
        "outliers": _df_to_records(out_df) if not out_df.empty else [],
        "heatmap": heatmap,
        "available_yield_classes": list(gm["Yield_Class"].unique()),
    }


FEATURE_CATEGORY_MAP: dict[str, str] = {
    "NDVI": "Greenness", "SAVI": "Greenness", "EVI": "Greenness",
    "CIgreen": "Chlorophyll", "MTCI": "Chlorophyll", "NDCI": "Chlorophyll", "NDRE": "Chlorophyll",
    "MSI": "Water Status", "NDWI": "Water Status", "WI": "Water Status",
    "EXG": "Canopy Color", "fractional_cover": "Canopy Structure",
    "average_height": "Structural", "ALS": "Senescence Dynamics",
    "PSRI": "Senescence Dynamics", "SIPI": "Pigment",
    "WALS": "Senescence Dynamics", "LPT_70": "Structural",
}


def get_category_summary() -> dict:
    """4L – Category-level outlier summary."""
    outlier_data = get_outliers()
    if not outlier_data["outliers"]:
        return {"category_summary": [], "heatmap_matrix": []}

    out_df = pd.DataFrame(outlier_data["outliers"])
    out_df["Feature_Category"] = out_df["Feature"].apply(
        lambda f: next((v for k, v in FEATURE_CATEGORY_MAP.items() if k.lower() in f.lower()), "Other")
    )
    cat_sum = (
        out_df.groupby(["Yield_Class", "Feature_Category"])
        .agg(
            Num_Genotypes=("genotype", "nunique"),
            Num_Features=("Feature", "nunique"),
            Mean_Abs_Z=("Z_score", lambda x: float(np.mean(np.abs(x)))),
        )
        .reset_index()
        .sort_values(["Yield_Class", "Num_Genotypes"], ascending=[True, False])
    )
    hm = (
        out_df.groupby(["Feature_Category", "Yield_Class"])["Z_score"]
        .apply(lambda x: float(np.mean(np.abs(x))))
        .reset_index()
        .rename(columns={"Z_score": "Mean_Abs_Z"})
    )
    return {
        "category_summary": _df_to_records(cat_sum),
        "heatmap_matrix": _df_to_records(hm),
    }


def get_chat_context() -> dict:
    """Compact summary for GPT chat prompts."""
    df = _load_temporal()
    _, gm, t33, t66 = _prepare(df)

    top_yield = (
        gm[["genotype", "Yield", "Yield_Class"]]
        .sort_values("Yield", ascending=False)
        .head(5)
    )

    stability = get_stability()
    cat_counts = {c["cv_category"]: c["count"] for c in stability.get("category_counts", [])}

    water_features = [
        ("MSI_mean", "MSI"),
        ("NDWI_mean", "NDWI"),
        ("WI_mean", "WI"),
    ]
    correlations = []
    for col, label in water_features:
        if col not in gm.columns:
            continue
        sub = gm[["Yield", col]].dropna()
        if len(sub) < 5:
            continue
        try:
            r_p, _ = pearsonr(sub["Yield"], sub[col])
        except Exception:
            continue
        correlations.append(f"{label} r={_safe_float(r_p)}")

    top_lines = [
        f"Genotype {int(r['genotype'])}: Yield={_safe_float(r['Yield'])}, Class={r['Yield_Class']}"
        for _, r in top_yield.iterrows()
    ]

    context = "\n".join(
        [
            "Temporal summary (auto):",
            f"Yield class thresholds: t33={_safe_float(t33)}, t66={_safe_float(t66)}",
            "Stability counts (CV): "
            f"Low={cat_counts.get('Low variation (<10%)', 0)}, "
            f"Moderate={cat_counts.get('Moderate variation (10–25%)', 0)}, "
            f"High={cat_counts.get('High variation (>25%)', 0)}",
            "Top 5 genotypes by yield:",
            *top_lines,
            "Water status correlations with Yield (Pearson r): " + (", ".join(correlations) or "N/A"),
            "Note: NDWI/WI and MSI are water status indices; interpret with domain context.",
        ]
    )

    return {"context": context}

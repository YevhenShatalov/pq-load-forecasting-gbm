#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob, itertools, warnings, hashlib
from typing import Dict, Tuple, List, Optional, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- External DM test (pip install dieboldmariano) ----
try:
    # API: dm_test(actual, pred1, pred2, loss=callable, h=..., one_sided=False,
    #              harvey_correction=True, variance_estimator="bartlett")
    from dieboldmariano import dm_test as dm_lib_test
except Exception as e:
    raise SystemExit(
        "Missing dependency 'dieboldmariano'. Install with: pip install dieboldmariano"
    ) from e

# ---------------- Configuration ----------------
INPUT_DIR  = rf"Evaluated"        # <<<<<< CHANGE THIS
OUTPUT_DIR = rf"Evaluated\DM"      # <<<<<< CHANGE THIS

H = 24                 # forecast horizon in steps (e.g., 24 for day-ahead hourly)
SPLIT_BY_EXP = False   # if True, treat each "exp" inside a file as a distinct model
MIN_OVERLAP = 5        # minimum aligned points to run DM for a pair

# ---- NEW: visibility gate for 80/95% intervals on forecast plots ----
INTERVAL_MIN_REL_WIDTH = 0.02
INTERVAL_REF_PERCENTILES = (5, 95)
INTERVAL_REF_SERIES = "y_true"          # 'y_true' or 'y_pred'

# ---- NEW: error metric knobs ----
MASE_SEASONAL_PERIOD = 24               # seasonal naive lag (e.g., 24 for hourly day-ahead)
MAPE_SAFE_EPS = 1e-8                    # drop points with |y| < this in MAPE
WMAPE_SAFE_EPS = 1e-8                   # avoid division by ~0 in WMAPE
SMAPE_SAFE_EPS = 1e-8                   # epsilon in |y|+|ŷ| denominator

# ------------------------------------------------

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

def _sanitize_id(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '|', str(s)).strip('|')

def _to_num(x):
    return pd.to_numeric(x, errors="coerce")

# --------- Excel reading & normalization ----------
def _pick_sheet(xl: pd.ExcelFile, target_tag: str) -> Optional[str]:
    # prioritize "*prediction*" then fallback to any sheet containing target tag
    for s in xl.sheet_names:
        low = s.lower()
        if target_tag.lower() in low and ("predict" in low or "pred" in low):
            return s
    for s in xl.sheet_names:
        if target_tag.lower() in s.lower():
            return s
    # last resort: if exactly two sheets and one is "*daily*", pick the other
    if len(xl.sheet_names) == 2:
        cands = [s for s in xl.sheet_names if "daily" not in s.lower()]
        if cands: return cands[0]
    return None

def _normalize_pred_df(df: pd.DataFrame) -> pd.DataFrame:
    # standardize to ts, y_true, y_pred; coerce numerics; keep optional interval/quantile/pinball/CRPS
    df = df.copy()
    # ts column
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])
    elif "Datetime" in df.columns:
        df = df.rename(columns={"Datetime":"ts"})
        df["ts"] = pd.to_datetime(df["ts"])
    else:
        df.iloc[:,0] = pd.to_datetime(df.iloc[:,0])
        df = df.rename(columns={df.columns[0]:"ts"})
    # normalize y_true/y_pred naming
    lower = {c.lower(): c for c in df.columns}
    if "y_true" not in lower or "y_pred" not in lower:
        for c in list(df.columns):
            cl = c.strip().lower()
            if cl in ("true","actual","target","y","obs","value"):
                df = df.rename(columns={c:"y_true"})
            if cl in ("pred","prediction","forecast","yhat","y_pred"):
                df = df.rename(columns={c:"y_pred"})
    if not {"y_true","y_pred"}.issubset(df.columns):
        missing = {"y_true","y_pred"} - set(df.columns)
        raise ValueError(f"Required columns missing {missing}. Found columns: {list(df.columns)[:15]}...")
    # numerics
    for c in ["y_true","y_pred","L80","U80","L95","U95","CRPS"]:
        if c in df.columns: df[c] = _to_num(df[c])
    # date helper
    if "date" in df.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                df["date"] = pd.to_datetime(df["date"]).dt.date
            except Exception:
                df["date"] = pd.to_datetime(df["ts"]).dt.date
    else:
        df["date"] = pd.to_datetime(df["ts"]).dt.date
    # sort
    df = df.sort_values("ts").reset_index(drop=True)
    return df

def read_model_file(path: str) -> Dict[str, pd.DataFrame]:
    """
    Returns dict like {'P_Power': dfP, 'Q_Power': dfQ} for the file, if present.
    """
    out: Dict[str, pd.DataFrame] = {}
    xl = pd.ExcelFile(path)
    for tgt in ("P_Power","Q_Power"):
        sname = _pick_sheet(xl, tgt)
        if sname is None:
            continue
        df = xl.parse(sname)
        df = _normalize_pred_df(df)
        out[tgt] = df
    return out

# ---------- Pinball & CRPS helpers ----------
def _pinball_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if str(c).lower().startswith("pinball_")]

_Q_PAT = re.compile(r"^q(?P<nn>\d{1,2}(\.\d+)?)$", re.IGNORECASE)  # Q05, Q5, Q95, Q10.5, Q0.05

def _quantile_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        cl = str(c).strip().lower()
        if _Q_PAT.match(cl):
            cols.append(c)
    return sorted(set(cols), key=lambda x: str(x).lower())

def _tau_from_quantile_name(name: str) -> Optional[float]:
    m = _Q_PAT.match(str(name).strip().lower())
    if not m:
        return None
    nn = m.group("nn")
    try:
        v = float(nn)
        return v/100.0 if v > 1 else v
    except Exception:
        return None

def _pinball_from_quantiles_row(y: float, taus: Sequence[float], qvals: Sequence[float]) -> float:
    """Average pinball across available taus for a single row (uses same taus order as qvals)."""
    acc = []
    for tau, qv in zip(taus, qvals):
        if pd.isna(y) or pd.isna(qv): 
            continue
        diff = y - qv
        # standard pinball loss ρ_τ(y, q) = (τ - 1{y<q}) * (y - q)
        loss = (tau - (diff < 0)) * diff
        acc.append(loss)
    return float(np.mean(acc)) if acc else np.nan

def _crps_from_quantiles_row(y: float, taus: Sequence[float], qvals: Sequence[float]) -> float:
    """Discrete approximation to CRPS via quantile integral: 2∫_0^1 ρ_τ(y, q_τ) dτ (trapezoid)."""
    pts = []
    for tau, qv in zip(taus, qvals):
        if pd.isna(y) or pd.isna(qv): 
            continue
        diff = y - qv
        loss = (tau - (diff < 0)) * diff
        pts.append((tau, loss))
    if not pts:
        return np.nan
    pts.sort(key=lambda t: t[0])
    taus_s = np.array([t for t, _ in pts], dtype=float)
    vals_s = np.array([v for _, v in pts], dtype=float)
    return 2.0 * float(np.trapz(vals_s, taus_s))

# ---------- Plotting helpers ----------
def _decorate_axes(title: str, xlabel: str, ylabel: str, legend: bool = True):
    """Apply consistent formatting: title/labels, minor ticks, dotted grid, legend."""
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.minorticks_on()  # minor ticks OFF by default; enable so minor grid shows
    plt.grid(True, which="both", axis="both", linestyle=":", linewidth=0.6, alpha=0.7)
    if legend:
        plt.legend(loc="best", fontsize=9, frameon=True)

def _robust_range(x: np.ndarray, p_lo: float, p_hi: float) -> float:
    """Robust range = percentile_hi - percentile_lo, ignoring NaNs."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    lo = np.nanpercentile(x, p_lo)
    hi = np.nanpercentile(x, p_hi)
    rng = float(hi - lo)
    if not np.isfinite(rng) or rng <= 0:
        rng = float(np.nanmax(x) - np.nanmin(x))
    return rng

def _band_is_visible(df: pd.DataFrame, low_col: str, high_col: str,
                     ref_col: str, rel_threshold: float,
                     ref_pcts: Tuple[float, float]) -> bool:
    """
    Decide if band [low_col, high_col] is wide enough to plot.
    Criterion: median(U-L) / robust_range(ref) >= rel_threshold.
    """
    if not {low_col, high_col, ref_col}.issubset(df.columns):
        return False
    L = _to_num(df[low_col]).to_numpy(dtype=float)
    U = _to_num(df[high_col]).to_numpy(dtype=float)
    W = U - L
    med_w = float(np.nanmedian(W)) if np.isfinite(W).any() else np.nan
    if not np.isfinite(med_w) or med_w <= 0:
        return False
    ref = _to_num(df[ref_col]).to_numpy(dtype=float)
    rng = _robust_range(ref, ref_pcts[0], ref_pcts[1])
    if not np.isfinite(rng) or rng <= 0:
        return True  # cannot assess scale reliably → draw
    return (med_w / rng) >= rel_threshold

def plot_forecast_with_intervals(df: pd.DataFrame, title: str, out_png: str):
    plt.figure(figsize=(11,5))
    plt.plot(df["ts"], df["y_true"], label="True", linewidth=1.2)
    plt.plot(df["ts"], df["y_pred"], label="Predicted", linewidth=1.2)

    # --- NEW: show 95/80 bands only if wide enough ---
    ref_col = INTERVAL_REF_SERIES if INTERVAL_REF_SERIES in df.columns else "y_true"
    show95 = {"L95","U95"}.issubset(df.columns) and _band_is_visible(
        df, "L95", "U95", ref_col, INTERVAL_MIN_REL_WIDTH, INTERVAL_REF_PERCENTILES
    )
    show80 = {"L80","U80"}.issubset(df.columns) and _band_is_visible(
        df, "L80", "U80", ref_col, INTERVAL_MIN_REL_WIDTH, INTERVAL_REF_PERCENTILES
    )

    if show95:
        plt.fill_between(df["ts"], df["L95"].astype(float).values, df["U95"].astype(float).values,
                         alpha=0.18, label="95% interval")
    if show80:
        plt.fill_between(df["ts"], df["L80"].astype(float).values, df["U80"].astype(float).values,
                         alpha=0.30, label="80% interval")

    _decorate_axes(title, "Time", "Value", legend=True)
    _ensure_dir(os.path.dirname(out_png))
    plt.tight_layout()
    plt.savefig(out_png, dpi=160); plt.close()
    return out_png

def plot_daily_pinball_and_crps(df: pd.DataFrame, title: str, out_png: str):
    g = df.copy()
    pin_cols = _pinball_columns(g)
    q_cols   = _quantile_columns(g)
    taus = [ _tau_from_quantile_name(c) for c in q_cols ]
    taus = [t for t in taus if t is not None]

    if pin_cols:
        g["Pinball_mean"] = g[pin_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    elif q_cols and taus:
        qvals = g[q_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        g["Pinball_mean"] = [
            _pinball_from_quantiles_row(y, taus, row) for y, row in zip(g["y_true"].astype(float), qvals)
        ]
    else:
        g["Pinball_mean"] = np.nan

    if "CRPS" in g.columns:
        g["CRPS"] = _to_num(g["CRPS"])
    elif q_cols and taus:
        qvals = g[q_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        g["CRPS"] = [
            _crps_from_quantiles_row(y, taus, row) for y, row in zip(g["y_true"].astype(float), qvals)
        ]

    if "date" not in g.columns:
        g["date"] = pd.to_datetime(g["ts"]).dt.date

    daily = g.groupby("date", as_index=False).agg({"Pinball_mean":"mean", "CRPS":"mean"})
    plt.figure(figsize=(11,5))
    plt.plot(pd.to_datetime(daily["date"]), daily["Pinball_mean"], label="Mean Pinball (daily)", linewidth=1.2)
    if "CRPS" in daily.columns:
        plt.plot(pd.to_datetime(daily["date"]), daily["CRPS"], label="CRPS (daily mean)", linewidth=1.2)
    _decorate_axes(title, "Date", "Score", legend=True)
    _ensure_dir(os.path.dirname(out_png))
    plt.tight_layout()
    plt.savefig(out_png, dpi=160); plt.close()
    return out_png

# ---------- DM across models (pairwise alignment) ----------
def _pairwise_align_predictions(A: pd.DataFrame, B: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise align by timestamp; keep y_true from the *left* frame only.
    Comparisons are based on model forecasts vs a single actual series.
    """
    M = A[["ts","y_true","y_pred"]].merge(B[["ts","y_pred"]], on="ts", suffixes=("_A","_B"))
    M.rename(columns={"y_true":"y", "y_pred_A":"p_i", "y_pred_B":"p_j"}, inplace=True)
    m = np.isfinite(_to_num(M["y"])) & np.isfinite(_to_num(M["p_i"])) & np.isfinite(_to_num(M["p_j"]))
    return M.loc[m, ["ts","y","p_i","p_j"]]

def dm_pairwise_matrix(models: Dict[str, pd.DataFrame], *, h: Optional[int] = 1):
    """
    Pairwise Diebold–Mariano matrices across models for a single target.
    Uses dieboldmariano.dm_test on forecast values ONLY (squared-error loss).
    """
    ids = sorted(models.keys())
    stat = pd.DataFrame(np.nan, index=ids, columns=ids, dtype=float)
    pval = stat.copy()
    nn   = stat.copy()

    # pointwise squared-error loss
    loss_fn = lambda y, yhat: (y - yhat) ** 2

    for i, j in itertools.product(ids, ids):
        if i == j:
            stat.loc[i, j] = 0.0
            pval.loc[i, j] = 1.0
            nn .loc[i, j]  = len(models[i])
            continue

        M = _pairwise_align_predictions(models[i], models[j])
        y, p_i, p_j = M["y"].to_numpy(float), M["p_i"].to_numpy(float), M["p_j"].to_numpy(float)
        nn.loc[i, j] = int(y.size)
        if y.size < MIN_OVERLAP:
            continue  # insufficient overlap → leave NaNs

        try:
            dm_stat, dm_p = dm_lib_test(
                y, p_i, p_j,
                loss=loss_fn,
                h=int(h) if h is not None else 1,
                one_sided=False,
                harvey_correction=True,
                variance_estimator="bartlett"
            )
        except Exception as ex:
            print(f"[WARN] DM failed for pair ({i}, {j}): {ex}")
            continue

        stat.loc[i, j] = float(dm_stat)
        pval.loc[i, j] = float(dm_p)

    return stat, pval, nn

# ---------- Orchestration ----------
def select_single_variant(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    """If exp exists with multiple values, pick the exp with the lowest MAE (only y_true/y_pred)."""
    if "exp" in df.columns and df["exp"].nunique() > 1:
        scores = []
        for exp, g in df.groupby("exp"):
            ae = (_to_num(g["y_true"]) - _to_num(g["y_pred"])).abs()
            mae = float(np.nanmean(ae))
            scores.append((mae, str(exp)))
        scores.sort()
        chosen = scores[0][1]
        out = df[df["exp"].astype(str) == chosen].copy().reset_index(drop=True)
        return out, chosen
    return df, None

def load_all_models(input_dir: str):
    files = [p for p in glob.glob(os.path.join(input_dir, "*.xlsx")) if not os.path.basename(p).startswith("~$")]
    models_by_target: Dict[str, Dict[str, pd.DataFrame]] = {"P_Power":{}, "Q_Power":{}}
    meta_rows = []
    for p in files:
        try:
            data = read_model_file(p)
        except Exception as ex:
            print(f"[WARN] Skipping {p}: {ex}")
            continue
        model_base = os.path.splitext(os.path.basename(p))[0]
        for tgt, df in data.items():
            if SPLIT_BY_EXP and ("exp" in df.columns) and (df["exp"].nunique() > 1):
                for exp, g in df.groupby("exp"):
                    m_id = _sanitize_id(f"{model_base}|{exp}")
                    models_by_target[tgt][m_id] = g.copy().reset_index(drop=True)
                    meta_rows.append({"target":tgt, "file":p, "model_id":m_id, "chosen_exp":str(exp)})
            else:
                chosen_exp = None
                if ("exp" in df.columns) and (df["exp"].nunique() > 1):
                    df, chosen_exp = select_single_variant(df)
                m_id = _sanitize_id(model_base)
                models_by_target[tgt][m_id] = df
                meta_rows.append({"target":tgt, "file":p, "model_id":m_id, "chosen_exp":chosen_exp})
    meta = pd.DataFrame(meta_rows)
    return models_by_target, meta

# --- Family/variant parsing + style maps for global daily plots ---
def _parse_family_variant(mid: str) -> Tuple[str, str]:
    """
    Parse a model id into (family, variant). Examples:
      'mutual_lags_LGBM_sheet2_83c06714_PQ_Power_best' -> ('mutual_lags_LGBM','sheet2')
      'SFS_shared_LGBM_sheet0_5cd3c8b3_PQ_Power_best' -> ('SFS_shared_LGBM','sheet0')
    Robust extras: seed#, fold#, v#, exp#, run#. Fallback: stable hash bucket.
    """
    name = str(mid)

    # strip common trailing tags
    name = re.sub(r'_(?:PQ|P|Q)_Power_best$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'_[0-9a-fA-F]{6,}$', '', name)

    # try explicit 'sheet#'
    m = re.search(r'^(?P<family>.+?)_sheet(?P<num>\d+)\b', name, flags=re.IGNORECASE)
    if m:
        return m.group('family'), f"sheet{m.group('num')}"

    # try other common variant tags
    m2 = re.search(r'^(?P<family>.+?)_(?P<tag>(?:seed|fold|exp|run|v))(?P<num>\d+)\b', name, flags=re.IGNORECASE)
    if m2:
        return m2.group('family'), f"{m2.group('tag').lower()}{m2.group('num')}"

    # fallback split, then stable hash
    parts = name.rsplit('_', 1)
    family = parts[0] if len(parts) > 1 else name
    tail = parts[1] if len(parts) > 1 else "base"
    # stable small hash suffix so same tail → same style
    h = hashlib.sha1(tail.encode('utf-8')).hexdigest()[:3]
    return family, f"v{h}"

def _family_color_map(families: List[str]) -> Dict[str, tuple]:
    """
    Stable color per family using a categorical palette ('tab20').
    """
    cmap = plt.get_cmap('tab20')  # qualitative colormap, suited for many discrete groups
    if hasattr(cmap, 'colors'):
        base_colors = list(cmap.colors)
    else:
        base_colors = [cmap(i / max(1, len(families))) for i in range(len(families))]
    color_map = {}
    for idx, fam in enumerate(sorted(families)):
        color_map[fam] = base_colors[idx % len(base_colors)]
    return color_map

# A larger bank of distinctive dash patterns (cycled per family)
_DASH_BANK: List[tuple] = [
    '-', '--', '-.', ':',
    (0, (5, 1)),           # long dash, short gap
    (0, (3, 1, 1, 1)),     # dash-dot-dot
    (0, (5, 2, 1, 2)),     # long dash, gap, short dash, gap
    (0, (1, 1)),           # dotted
    (0, (7, 2)),           # extra long dash
    (0, (2, 1, 2, 1, 2, 1)),  # triple dash
    (0, (4, 1, 1, 1)),     # dash + tiny gaps
    (0, (8, 1, 1, 1)),     # very long + dots
]

def _build_style_map(items: List[Dict]) -> Dict[Tuple[str, str], tuple]:
    """
    Build {(family, variant) -> linestyle} so that, within each family, variants
    get distinct dash styles from _DASH_BANK deterministically (sorted by variant).
    """
    fam2vars: Dict[str, List[str]] = {}
    for it in items:
        fam2vars.setdefault(it["family"], []).append(str(it["variant"]))
    style_map: Dict[Tuple[str, str], tuple] = {}
    for fam, variants in fam2vars.items():
        uniq = sorted(set(variants))
        for i, v in enumerate(uniq):
            style_map[(fam, v)] = _DASH_BANK[i % len(_DASH_BANK)]
    return style_map

# --- Global *daily* Pinball & CRPS plots (P/Q) with improved styles ---
def _compute_daily_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: ['date','Pinball_mean','CRPS'] (daily averages).
    """
    g = df.copy()
    pin_cols = _pinball_columns(g)
    q_cols   = _quantile_columns(g)
    taus = [ _tau_from_quantile_name(c) for c in q_cols ]
    taus = [t for t in taus if t is not None]

    if pin_cols:
        g["__pin__"] = g[pin_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    elif q_cols and taus:
        qvals = g[q_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        g["__pin__"] = [ _pinball_from_quantiles_row(y, taus, row) for y, row in zip(g["y_true"].astype(float), qvals) ]
    else:
        g["__pin__"] = np.nan

    if "CRPS" in g.columns:
        g["__crps__"] = _to_num(g["CRPS"])
    elif q_cols and taus:
        qvals = g[q_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        g["__crps__"] = [ _crps_from_quantiles_row(y, taus, row) for y, row in zip(g["y_true"].astype(float), qvals) ]
    else:
        g["__crps__"] = np.nan

    if "date" not in g.columns:
        g["date"] = pd.to_datetime(g["ts"]).dt.date

    daily = (
        g.groupby("date", as_index=False)[["__pin__","__crps__"]]
         .mean()
         .rename(columns={"__pin__":"Pinball_mean","__crps__":"CRPS"})
    )
    return daily

def build_global_daily_trend_plots(models_by_target: Dict[str, Dict[str, pd.DataFrame]], out_dir: str):
    """
    Creates four line charts, split by target:
       - global_daily_Pinball_P.png, global_daily_CRPS_P.png
       - global_daily_Pinball_Q.png, global_daily_CRPS_Q.png

    Each model gets a line: **color = family**, **linestyle = per-family variant dash**.
    Colors/styles are consistent across P and Q by computing the maps once across both.
    """
    items_by_target = {"P": [], "Q": []}  # each item: dict(label, series_pin, series_crps, family, variant)
    all_items: List[Dict] = []

    for tgt, models in models_by_target.items():
        t_tag = "P" if tgt.startswith("P") else "Q"
        for mid, df in models.items():
            family, variant = _parse_family_variant(mid)
            label = mid
            daily = _compute_daily_scores(df)
            if daily.empty:
                continue
            idx = pd.to_datetime(daily["date"])
            series_pin  = pd.Series(daily["Pinball_mean"].to_numpy(float), index=idx).sort_index()
            series_crps = pd.Series(daily["CRPS"].to_numpy(float),          index=idx).sort_index()
            rec = {"label": label, "series_pin": series_pin, "series_crps": series_crps,
                   "family": family, "variant": variant}
            items_by_target[t_tag].append(rec)
            all_items.append(rec)

    if not items_by_target["P"] and not items_by_target["Q"]:
        return

    _ensure_dir(out_dir)

    families = sorted({it["family"] for it in all_items})
    fam2color = _family_color_map(families)
    style_map = _build_style_map(all_items)

    def _plot_trends(items: List[Dict], series_key: str, title: str, ylabel: str, out_name: str):
        if not items:
            return
        # unified x-axis within this target
        all_idx = sorted(set().union(*[it[series_key].index for it in items]))
        plt.figure(figsize=(13, 6))
        for it in items:
            s_aligned = it[series_key].reindex(all_idx)
            color = fam2color[it["family"]]
            linestyle = style_map[(it["family"], str(it["variant"]))]
            plt.plot(all_idx, s_aligned.values,
                     label=it["label"], linewidth=1.2, alpha=0.95,
                     color=color, linestyle=linestyle)
        _decorate_axes(title, "Date", ylabel, legend=False)
        # Multi-column legend; 2 or 3 columns depending on crowding
        n_models = len(items)
        ncol = 2 if n_models <= 18 else 3
        leg = plt.legend(loc="upper left", ncol=ncol, fontsize=8, frameon=True)
        for line in leg.get_lines():
            line.set_linewidth(2.0)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, out_name), dpi=160)
        plt.close()

    # P-target figures
    _plot_trends(items_by_target["P"], "series_pin",
                 "Global daily Pinball — P models (per day)",
                 "Daily mean Pinball", "global_daily_Pinball_P.png")
    _plot_trends(items_by_target["P"], "series_crps",
                 "Global daily CRPS — P models (per day)",
                 "Daily mean CRPS", "global_daily_CRPS_P.png")

    # Q-target figures
    _plot_trends(items_by_target["Q"], "series_pin",
                 "Global daily Pinball — Q models (per day)",
                 "Daily mean Pinball", "global_daily_Pinball_Q.png")
    _plot_trends(items_by_target["Q"], "series_crps",
                 "Global daily CRPS — Q models (per day)",
                 "Daily mean CRPS", "global_daily_CRPS_Q.png")

# --- NEW: global daily classical error metric plots (RMSE, MAE, MAPE%, SMAPE%, WMAPE%, MASE) ---
def _mase_denominator(y_true: np.ndarray, s: int) -> float:
    """MASE denominator = mean absolute seasonal naive difference |y_t - y_{t-s}| across the series."""
    y = np.asarray(y_true, dtype=float)
    if not np.isfinite(y).any() or s is None or s <= 0 or y.size <= s:
        return np.nan
    y1 = y[s:]
    y0 = y[:-s]
    m  = np.isfinite(y1) & np.isfinite(y0)
    diffs = np.abs(y1[m] - y0[m])
    return float(np.nanmean(diffs)) if diffs.size else np.nan

def _compute_daily_error_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-day metrics:
      ['date','RMSE','MAE','MAPE%','SMAPE%','WMAPE%','MASE']
    """
    g = df.copy()
    g["y_true"] = _to_num(g["y_true"])
    g["y_pred"] = _to_num(g["y_pred"])
    if "date" not in g.columns:
        g["date"] = pd.to_datetime(g["ts"]).dt.date

    # errors
    err = (g["y_pred"] - g["y_true"]).to_numpy(float)
    abs_y = np.abs(g["y_true"].to_numpy(float))
    abs_e = np.abs(err)
    se    = err ** 2

    # per-row helpers (fractions; convert to % later)
    with np.errstate(divide='ignore', invalid='ignore'):
        ape = np.where(abs_y > MAPE_SAFE_EPS, abs_e / abs_y, np.nan)  # fraction
        smape_frac = 2.0 * abs_e / (abs_y + np.abs(g["y_pred"].to_numpy(float)) + SMAPE_SAFE_EPS)

    g["_SE"] = se
    g["_AE"] = abs_e
    g["_ABSY"] = abs_y
    g["_APE"] = ape
    g["_SMAPE_FRAC"] = smape_frac

    # daily aggregates
    daily = (g.groupby("date", as_index=False)
               .agg(SE_mean=("_SE","mean"),
                    MAE=("_AE","mean"),
                    APE_mean=("_APE","mean"),
                    SMAPE_frac_mean=("_SMAPE_FRAC","mean"),
                    AE_sum=("_AE","sum"),
                    ABSY_sum=("_ABSY","sum")))

    # RMSE
    daily["RMSE"] = np.sqrt(daily["SE_mean"])

    # MAPE, SMAPE, WMAPE as percents
    daily["MAPE%"]   = 100.0 * daily["APE_mean"]
    daily["SMAPE%"]  = 100.0 * daily["SMAPE_frac_mean"]
    daily["WMAPE%"]  = 100.0 * np.where(daily["ABSY_sum"] > WMAPE_SAFE_EPS,
                                        daily["AE_sum"] / daily["ABSY_sum"], np.nan)

    # MASE denominator from entire series (constant per model)
    mase_denom = _mase_denominator(g["y_true"].to_numpy(float), s=int(MASE_SEASONAL_PERIOD))
    daily["MASE"] = np.where(np.isfinite(mase_denom) and mase_denom > 0.0,
                             daily["MAE"] / mase_denom, np.nan)

    # final selection
    out = daily[["date","RMSE","MAE","MAPE%","SMAPE%","WMAPE%","MASE"]].copy()
    return out

_METRIC_DISPLAY: List[Tuple[str, str, str]] = [
    ("RMSE",    "RMSE",        "Daily RMSE"),
    ("MAE",     "MAE",         "Daily MAE"),
    ("MAPE%",   "MAPE (%)",    "Daily MAPE (%)"),
    ("SMAPE%",  "SMAPE (%)",   "Daily SMAPE (%)"),
    ("WMAPE%",  "WMAPE (%)",   "Daily WMAPE (%)"),
    ("MASE",    "MASE",        "Daily MASE"),
]

def _safe_metric_filename_key(metric_key: str) -> str:
    return metric_key.replace("%", "pct").replace(" ", "")

def build_global_daily_metric_plots(models_by_target: Dict[str, Dict[str, pd.DataFrame]], out_dir: str):
    """
    Builds daily overlay plots for classical error metrics (RMSE, MAE, MAPE%, SMAPE%, WMAPE%, MASE)
    separately for P and Q targets. Each model gets a line; **color by family** and
    **dash style by variant within family**.
    """
    items_by_target = {"P": [], "Q": []}
    all_items: List[Dict] = []

    for tgt, models in models_by_target.items():
        t_tag = "P" if tgt.startswith("P") else "Q"
        for mid, df in models.items():
            family, variant = _parse_family_variant(mid)
            label = mid
            daily = _compute_daily_error_metrics(df)
            if daily.empty:
                continue
            idx = pd.to_datetime(daily["date"])
            series_by_metric = {
                k: pd.Series(daily[k].to_numpy(float), index=idx).sort_index()
                for (k, _, _) in _METRIC_DISPLAY
                if k in daily.columns
            }
            rec = {"label": label, "family": family, "variant": variant, "series_by_metric": series_by_metric}
            items_by_target[t_tag].append(rec)
            all_items.append(rec)

    if not items_by_target["P"] and not items_by_target["Q"]:
        return

    _ensure_dir(out_dir)
    families = sorted({it["family"] for it in all_items})
    fam2color = _family_color_map(families)
    style_map = _build_style_map(all_items)

    def _plot_metric(items: List[Dict], metric_key: str, display_name: str, ylabel: str, out_name: str):
        # collect non-empty series for this metric
        usable = [it for it in items if metric_key in it["series_by_metric"] and not it["series_by_metric"][metric_key].empty]
        if not usable:
            return
        all_idx = sorted(set().union(*[it["series_by_metric"][metric_key].index for it in usable]))
        plt.figure(figsize=(13, 6))
        for it in usable:
            s_aligned = it["series_by_metric"][metric_key].reindex(all_idx)
            color = fam2color[it["family"]]
            linestyle = style_map[(it["family"], str(it["variant"]))]
            plt.plot(all_idx, s_aligned.values,
                     label=it["label"], linewidth=1.2, alpha=0.95,
                     color=color, linestyle=linestyle)
        _decorate_axes(f"Global daily {display_name} — {'P' if items is items_by_target['P'] else 'Q'} models (per day)",
                       "Date", ylabel, legend=False)
        n_models = len(usable)
        ncol = 2 if n_models <= 18 else 3
        leg = plt.legend(loc="upper left", ncol=ncol, fontsize=8, frameon=True)
        for line in leg.get_lines():
            line.set_linewidth(2.0)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, out_name), dpi=160)
        plt.close()

    # For each target and each metric, render
    for t_tag, items in items_by_target.items():
        for metric_key, display_name, ylabel in _METRIC_DISPLAY:
            fname_key = _safe_metric_filename_key(metric_key)
            out_name = f"global_daily_{fname_key}_{t_tag}.png"
            _plot_metric(items, metric_key, display_name, ylabel, out_name)

def main():
    if not os.path.isdir(INPUT_DIR):
        raise SystemExit(f"INPUT_DIR does not exist: {INPUT_DIR}")
    _ensure_dir(OUTPUT_DIR)
    plot_dir = _ensure_dir(os.path.join(OUTPUT_DIR, "plots"))
    dm_xlsx = os.path.join(OUTPUT_DIR, "DM_across_models.xlsx")

    models_by_target, meta = load_all_models(INPUT_DIR)
    # save meta
    meta.to_csv(os.path.join(OUTPUT_DIR, "models_meta.csv"), index=False)

    # per-model plots
    for tgt, models in models_by_target.items():
        tdir = _ensure_dir(os.path.join(plot_dir, tgt))
        for mid, df in models.items():
            f1 = os.path.join(tdir, f"forecast_{tgt}_{mid}.png")
            f2 = os.path.join(tdir, f"scores_{tgt}_{mid}.png")
            plot_forecast_with_intervals(df, title=f"{tgt} — {mid}: Forecast + 80/95% intervals", out_png=f1)
            plot_daily_pinball_and_crps(df, title=f"{tgt} — {mid}: Daily mean Pinball & CRPS", out_png=f2)

    # four global *daily* trend plots (Pinball/CRPS × P/Q), now with improved per-family dash styles
    build_global_daily_trend_plots(models_by_target, out_dir=plot_dir)

    # NEW: twelve global *daily* classical error plots (6 metrics × P/Q)
    build_global_daily_metric_plots(models_by_target, out_dir=plot_dir)

    # DM matrices per target (single loss: squared error)
    writer = pd.ExcelWriter(dm_xlsx, engine="xlsxwriter")
    meta.to_excel(writer, sheet_name="models", index=False)
    for tgt, models in models_by_target.items():
        if len(models) < 2:
            print(f"[INFO] Target {tgt}: only {len(models)} model(s) found; DM skipped.")
            continue
        h_eff = int(H) if H is not None else 1
        stat, pval, nn = dm_pairwise_matrix(models, h=h_eff)
        stat.to_excel(writer, sheet_name=f"{tgt}_DM_stat")
        pval.to_excel(writer, sheet_name=f"{tgt}_DM_p")
        nn.to_excel(writer,  sheet_name=f"{tgt}_DM_n")
    writer.close()
    print(f"[OK] Wrote DM workbook: {dm_xlsx}")
    print(f"[OK] Plots under: {plot_dir}")

if __name__ == "__main__":
    main()

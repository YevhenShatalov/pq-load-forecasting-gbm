# -*- coding: utf-8 -*-

from __future__ import annotations
import os, re, math, json, warnings, hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, List, Sequence, Optional

import numpy as np
import pandas as pd
from pandas.api.types import (
    CategoricalDtype, is_object_dtype, is_bool_dtype, is_integer_dtype
)

# ========= CONFIG (EDIT THESE) =========
DATA_PATH   = rf"data\PQ.xlsx"  # PQ.xlsx path
MODELS_DIR  = rf"Models"          # folder with *_best.* and *_best.meta.json
SAVE_DIR    = Path(rf"Evaluated")   # where to save the Excel workbooks

# Rolling/Walkforward controls
WALKFORWARD_DAILY = True        # turn on daily refit/predict
DAY_AHEAD_HOURS   = 24          # 24h-ahead (one-day-ahead)
TRAIN_WINDOW_MODE = "expanding" # "expanding" (default) or "sliding"
SLIDING_DAYS      = None        # if TRAIN_WINDOW_MODE=="sliding", keep last N days

# Global desired windows (each sheet is **clipped** to these). Use None/"auto" for sheet's earliest timestamp.
TRAIN_START = None                # e.g., "2021-01-01 00:00" or None/"auto"
TRAIN_END   = "2021-12-31 23:00"
TEST_START  = "2022-01-01 00:00"
TEST_END    = "2022-02-28 23:00"

# --- Prediction intervals & probabilistic scores ---
INTERVAL_LEVELS = (0.80, 0.95)         # always write L80/U80 and L95/U95
ALL_INTERVAL_COVERAGES = tuple(np.round(np.arange(0.05, 1.00, 0.05), 2))  # 0.05..0.95 → write Lxx/Uxx
QUANTILE_GRID = tuple(np.round(np.arange(0.05, 1.00, 0.05), 2))           # 0.05..0.95 → write Qxx
BUILD_INTERVALS = True           # must be True for quantiles/CRPS/Pinball (we need simulated paths)
BLOCK_SIZE_PI   = 24             # fallback block size when not hour-matching
N_SIM_PI        = 1000           # number of simulated paths for empirical quantiles / CRPS

RANDOM_SEED        = 42
XGB_NATIVE_CATS    = True  # True only if tuned XGB used native categoricals
SEASONALITY        = 24    # for MASE scaling

# Known categorical columns in PQ.xlsx
KNOWN_CATS = [
    "Rainy","hour","weekday","season_idx","day_in_season","season_len",
    "is_holiday","is_day_before_hol","is_day_after_hol","is_weekend",
    "is_new_year","is_jan2","is_old_new_year","is_orthxmas","is_dec25"
]
# ======================================


# ---------- Optional packages ----------
_pkg = {}
def _try_import():
    global _pkg
    try:
        import lightgbm as lgb
        _pkg["LGBM"] = lgb
    except Exception:
        _pkg["LGBM"] = None
    try:
        import xgboost as xgb
        _pkg["XGB"] = xgb
    except Exception:
        _pkg["XGB"] = None
    try:
        from catboost import CatBoostRegressor
        _pkg["CatBoost"] = CatBoostRegressor
    except Exception:
        _pkg["CatBoost"] = None
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
        _pkg["HGBR"] = HistGradientBoostingRegressor
        _pkg["GBR"]  = GradientBoostingRegressor
    except Exception:
        _pkg["HGBR"] = None
        _pkg["GBR"]  = None
    try:
        import joblib
        _pkg["joblib"] = joblib
    except Exception:
        _pkg["joblib"] = None
_try_import()


# --------------------------- Excel writer / filenames ---------------------------
def choose_excel_engine() -> Optional[str]:
    """Return 'xlsxwriter' or 'openpyxl' if installed; None if neither."""
    try:
        import xlsxwriter  # noqa: F401
        return "xlsxwriter"
    except Exception:
        pass
    try:
        import openpyxl  # noqa: F401
        return "openpyxl"
    except Exception:
        pass
    return None

def _safe_sheet_name(name: str) -> str:
    # Excel sheet names: <=31 chars, no []:*?/\
    bad = '[]:*?/\\'
    s = ''.join(ch if ch not in bad else '_' for ch in name)
    return s[:31]

def ensure_unique_path(path: Path) -> Path:
    """If the path exists, append __v2/__v3/... until unique."""
    i = 2
    out = Path(path)
    while out.exists():
        out = path.with_name(f"{path.stem}__v{i}{path.suffix}")
        i += 1
    return out

def short_sig(*parts: str) -> str:
    """Short deterministic signature to avoid filename collisions."""
    s = "|".join(str(p) for p in parts)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


# --------------------------- Path resolver (fallback to /mnt/data if needed) ----
def resolve_paths():
    global DATA_PATH, MODELS_DIR

    # 1) allow environment overrides first
    DATA_PATH  = os.environ.get("PQ_XLSX", DATA_PATH)
    MODELS_DIR = os.environ.get("MODELS_DIR", MODELS_DIR)

    # 2) now check what exists
    data_ok   = os.path.exists(DATA_PATH)
    models_ok = os.path.exists(MODELS_DIR) and any(Path(MODELS_DIR).glob("*_best.meta.json"))

    # 3) fall back to /mnt/data if still missing
    if not data_ok and os.path.exists("/mnt/data/PQ.xlsx"):
        print("[paths] DATA_PATH not found → using /mnt/data/PQ.xlsx")
        DATA_PATH = "/mnt/data/PQ.xlsx"

    if not models_ok and os.path.exists("/mnt/data") and any(Path("/mnt/data").glob("*_best.meta.json")):
        print("[paths] MODELS_DIR not found or empty → using /mnt/data")
        MODELS_DIR = "/mnt/data"


# --------------------------- Utilities ---------------------------
def _parse_ts_any(val: Optional[str]) -> Optional[pd.Timestamp]:
    if val is None or str(val).strip().lower() in {"", "auto", "none"}:
        return None
    return pd.Timestamp(val).floor("h")

def rmse(a, b):
    e = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean(e**2)))
def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))
def mape(a, b, eps=1e-8):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return float(np.mean(np.abs(b - a) / (np.abs(a) + eps)) * 100.0)
def smape(a, b, eps=1e-8):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return float(np.mean(2*np.abs(b - a) / (np.abs(a) + np.abs(b) + eps)) * 100.0)
def wmape(a, b, eps=1e-8):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return float(np.sum(np.abs(b - a)) / (np.sum(np.abs(a)) + eps) * 100.0)

SEASONALITY = 24
def mase(a_true: np.ndarray, a_pred: np.ndarray, y_train: np.ndarray, m: int=SEASONALITY) -> float:
    y = np.asarray(y_train).ravel()
    if len(y) <= m:
        return np.nan
    denom = np.mean(np.abs(y[m:] - y[:-m]))
    if denom <= 0:
        return np.nan
    return float(np.mean(np.abs(np.asarray(a_true) - np.asarray(a_pred))) / denom)

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
    return {
        "n": int(len(y_true)),
        "RMSE": rmse(y_true, y_pred),
        "MAE":  mae(y_true, y_pred),
        "MAPE%": mape(y_true, y_pred),
        "SMAPE%": smape(y_true, y_pred),
        "WMAPE%": wmape(y_true, y_pred),
        "MASE": mase(y_true, y_pred, y_train, m=SEASONALITY),
    }

# ---- Pinball (quantile) loss & CRPS over ensemble draws ----
def pinball_loss(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    """
    Per-forecast pinball loss at level tau, vectorized.
    y, q are arrays of the same shape (timestamps).
    """
    u = y - q
    return np.maximum(tau*u, (tau - 1.0)*u)

def crps_from_sims(y: np.ndarray, sims: np.ndarray) -> np.ndarray:
    """
    Per-forecast CRPS from simulated draws (shape sims = [R, T]).
    Formula: 1/M sum |x - y| - 1/(2M^2) sum_{i,j} |x_i - x_j|
    Vectorized in columns (timestamps).
    """
    sims = np.asarray(sims, dtype=float)         # (M, T)
    y = np.asarray(y, dtype=float).ravel()       # (T,)
    M = sims.shape[0]
    # term1
    term1 = np.mean(np.abs(sims - y[None, :]), axis=0)  # (T,)
    # term2 via sorted draws (O(M log M) per column)
    xs = np.sort(sims, axis=0)                           # (M, T)
    # weights w_k = 2k - M - 1 for k=1..M  (1-indexed)
    k = np.arange(1, M+1, dtype=float).reshape(-1, 1)
    w = (2.0*k - M - 1.0)                                # (M, 1)
    S = np.sum(w * xs, axis=0)                           # (T,)
    term2 = S / (M**2)
    return term1 - term2                                  # (T,)


# --------- Feature parsing & recomputation ----------
_lag_pat   = re.compile(r"^(?P<tgt>P(?:_Power)?|Q(?:_Power)?)_lag_(?P<k>\d+)$", re.IGNORECASE)
_rmean_pat = re.compile(r"^(?P<tgt>P(?:_Power)?|Q(?:_Power)?)_rmean_(?P<w>\d+)$", re.IGNORECASE)
_rstd_pat  = re.compile(r"^(?P<tgt>P(?:_Power)?|Q(?:_Power)?)_rstd_(?P<w>\d+)$", re.IGNORECASE)

def iter_day_blocks_for_sheet(idx: pd.DatetimeIndex,
                              *,
                              train_start: Optional[str],
                              test_start: str,
                              test_end: str,
                              horizon_hours: int = 24):
    """Yield day-by-day {train_start, train_end, test_start, test_end, date} dicts clipped to this sheet."""
    s_te = pd.Timestamp(test_start).normalize()
    e_te = pd.Timestamp(test_end).normalize()
    imin, imax = idx.min(), idx.max()
    if s_te < imin: s_te = imin.normalize()
    if e_te > imax: e_te = imax.normalize()

    for d in pd.date_range(s_te, e_te, freq="D"):
        te_start = d
        te_end   = d + pd.Timedelta(hours=horizon_hours - 1)
        hard_end = min(imax, pd.Timestamp(test_end))
        if te_end > hard_end:
            break
        tr_end = te_start - pd.Timedelta(hours=1)
        if TRAIN_WINDOW_MODE.lower() == "sliding" and SLIDING_DAYS:
            tr_start = max(pd.Timestamp(train_start) if train_start else imin,
                           tr_end - pd.Timedelta(days=int(SLIDING_DAYS)) + pd.Timedelta(hours=1))
        else:
            tr_start = pd.Timestamp(train_start) if train_start else imin
        tr_start = max(tr_start, imin)
        tr_end   = min(tr_end,   imax)
        yield {
            "train_start": str(tr_start), "train_end": str(tr_end),
            "test_start": str(te_start),  "test_end": str(te_end),
            "date": te_start.date()
        }

def parse_feature(name: str):
    m = _lag_pat.match(name)
    if m: return ("lag", m.group("tgt"), int(m.group("k")))
    m = _rmean_pat.match(name)
    if m: return ("rmean", m.group("tgt"), int(m.group("w")))
    m = _rstd_pat.match(name)
    if m: return ("rstd", m.group("tgt"), int(m.group("w")))
    return (None, None, None)

def recompute_feature_at(ts: pd.Timestamp, name: str, yhist: Dict[str, pd.Series]) -> float:
    """Recompute lag/rolling feature at timestamp `ts` using mixed observed+predicted history."""
    typ, tgt, par = parse_feature(name)
    if typ is None:
        raise ValueError(f"{name} is not a recognized recomputable feature")
    s = yhist[tgt]
    k = int(par)
    if typ == "lag":
        src_ts = ts - pd.Timedelta(hours=k)
        return float(s.get(src_ts, np.nan))
    elif typ in ("rmean", "rstd"):
        window_end   = ts - pd.Timedelta(hours=1)
        window_start = ts - pd.Timedelta(hours=k)
        seg = s.loc[window_start:window_end]
        if len(seg) == 0:
            return np.nan
        return float(seg.mean() if typ == "rmean" else seg.std(ddof=0))
    return np.nan


# --------- Identify targets & feature classification ----------
def identify_targets(df: pd.DataFrame) -> Tuple[str,str]:
    for cand in [("P_Power","Q_Power"), ("P","Q")]:
        if all(c in df.columns for c in cand):
            return cand
    raise RuntimeError("Could not find targets (P_Power,Q_Power) or (P,Q).")

def classify_features(df: pd.DataFrame, known_categorical: Sequence[str]) -> Tuple[List[str],List[str]]:
    cat_cols, num_cols = [], []
    known = set(map(str, known_categorical or []))
    for c in df.columns:
        s = df[c]
        if (c in known) or isinstance(s.dtype, CategoricalDtype) or is_object_dtype(s) or is_bool_dtype(s):
            cat_cols.append(c)
        else:
            if is_integer_dtype(s) and s.nunique(dropna=True) <= max(24, int(0.02*len(s))):
                cat_cols.append(c)
            else:
                num_cols.append(c)
    return num_cols, cat_cols


# ------------------------- Model boilerplate -------------------------
def ensure_named_df(X, columns_hint: Optional[Sequence[str]] = None, like_est: Optional[object] = None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    X_arr = np.asarray(X)
    cols = None
    if columns_hint is not None:
        cols = list(columns_hint)
    elif like_est is not None and hasattr(like_est, "feature_names_in_"):
        cols = list(getattr(like_est, "feature_names_in_"))
    if cols is None:
        cols = [f"col_{i}" for i in range(X_arr.shape[1])]
    return pd.DataFrame(X_arr, columns=cols)

def align_to_columns(X: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return ensure_named_df(X).reindex(columns=list(columns))

def prepare_X_for_model(model_name: str, Xdf: pd.DataFrame, cat_cols: Sequence[str], xgb_native_categorical: bool=True):
    """
    Returns: X_prepared, trained_columns, cat_index_for_catboost, ohe_cols
    """
    model_name = (model_name or "").upper()
    Xdf = Xdf.copy()
    for c in cat_cols:
        if c in Xdf.columns and not isinstance(Xdf[c].dtype, CategoricalDtype):
            Xdf[c] = Xdf[c].astype("category")
    trained_cols = list(Xdf.columns); cat_idx = []; ohe_cols = []

    if model_name == "CATBOOST":
        for c in cat_cols:
            if c in Xdf.columns:
                Xdf[c] = Xdf[c].astype("string").fillna("__nan__").astype(object)
        return Xdf, trained_cols, [Xdf.columns.get_loc(c) for c in cat_cols if c in Xdf.columns], ohe_cols

    if model_name in {"HGBR","GBR"} or (model_name == "XGB" and not xgb_native_categorical):
        before = set(Xdf.columns)
        if cat_cols:
            Xdf = pd.get_dummies(Xdf, columns=[c for c in cat_cols if c in Xdf.columns], dummy_na=False)
        ohe_cols = [c for c in Xdf.columns if c not in before]
        trained_cols = list(Xdf.columns)
        return Xdf, trained_cols, [], ohe_cols

    return Xdf, trained_cols, [], ohe_cols

def freeze_categories(X: pd.DataFrame, cat_levels: Dict[str, List[str]]) -> pd.DataFrame:
    if not cat_levels:
        return X
    X = X.copy()
    for c, levels in cat_levels.items():
        if c in X.columns and isinstance(X[c].dtype, CategoricalDtype):
            X[c] = X[c].cat.set_categories(levels)
    return X

def safe_for_model(X: pd.DataFrame, model_name: str) -> pd.DataFrame:
    model = (model_name or "").upper()
    if model == "GBR":
        try:
            from sklearn.impute import SimpleImputer
            imp = SimpleImputer(strategy="median", keep_empty_features=True)
            X_imp = imp.fit_transform(X)
            X_imp = np.asarray(X_imp, dtype=float)
            X_imp[np.isnan(X_imp)] = 0.0
            return pd.DataFrame(X_imp, index=X.index, columns=X.columns)
        except Exception:
            Xf = X.fillna(X.median(numeric_only=True))
            Xf = Xf.fillna(0.0)
            return Xf
    return X

def build_estimator_from_params(model_name: str, params: Dict, use_gpu: bool=True):
    name = (model_name or "").upper()
    if name == "LGBM" and _pkg["LGBM"] is not None:
        lgb = _pkg["LGBM"]
        p = dict(params or {}); p.setdefault("verbose", -1)
        est = lgb.LGBMRegressor(**p)
        try:
            if use_gpu:
                for k in ("device", "device_type"):
                    try: est.set_params(**{k: "gpu"}); break
                    except Exception: pass
            else:
                for k in ("device", "device_type"):
                    try: est.set_params(**{k: "cpu"}); break
                    except Exception: pass
        except Exception:
            pass
        return est
    if name == "XGB" and _pkg["XGB"] is not None:
        xgb = _pkg["XGB"]
        p = dict(params or {})
        est = xgb.XGBRegressor(**p)
        try:
            est.set_params(enable_categorical=bool(XGB_NATIVE_CATS))
        except Exception:
            pass
        try:
            est.set_params(device=("cuda" if use_gpu else "cpu"))
        except Exception:
            try:
                est.set_params(tree_method=("gpu_hist" if use_gpu else "hist"))
                if use_gpu: est.set_params(predictor="gpu_predictor")
            except Exception:
                pass
        return est
    if name == "CATBOOST" and _pkg["CatBoost"] is not None:
        CatBoostRegressor = _pkg["CatBoost"]
        p = dict(params or {}); p.setdefault("verbose", False)
        if use_gpu: p.setdefault("task_type", "GPU")
        return CatBoostRegressor(**p)
    if name == "HGBR" and _pkg["HGBR"] is not None:
        return _pkg["HGBR"](**(params or {}))
    if name == "GBR" and _pkg["GBR"] is not None:
        return _pkg["GBR"](**(params or {}))
    raise RuntimeError(f"Requested model '{model_name}' not available in this environment.")


# ------------------------- Flexible time windows (per sheet) -------------------------
def build_split_for_sheet(idx: pd.DatetimeIndex,
                          train_start, train_end, test_start, test_end) -> Dict[str,str]:
    """Clip requested bounds to this sheet's index (different sheets start later due to lags)."""
    s_tr = _parse_ts_any(train_start); e_tr = _parse_ts_any(train_end)
    s_te = _parse_ts_any(test_start);  e_te = _parse_ts_any(test_end)

    imin = idx.min(); imax = idx.max()

    def _clip(ts, name):
        if ts is None: return None
        if ts < imin:
            warnings.warn(f"{name} < sheet start ({imin}); clipped.")
            return imin
        if ts > imax:
            warnings.warn(f"{name} > sheet end ({imax}); clipped.")
            return imax
        return ts

    s_tr = _clip(s_tr, "train_start") if s_tr is not None else imin
    e_tr = _clip(e_tr, "train_end")   if e_tr is not None else (imax - pd.Timedelta(hours=1))
    s_te = _clip(s_te, "test_start")  if s_te is not None else (e_tr + pd.Timedelta(hours=1))
    e_te = _clip(e_te, "test_end")    if e_te is not None else imax

    if not (s_tr <= e_tr < s_te <= e_te):
        raise RuntimeError(f"Bad split after clipping: train {s_tr}..{e_tr}; test {s_te}..{e_te}")
    return {
        "train_start": str(s_tr), "train_end": str(e_tr),
        "test_start":  str(s_te), "test_end":  str(e_te)
    }


# ------------------------- Read PQ (all sheets) -------------------------
def read_pq_xlsx(path: str) -> List[pd.DataFrame]:
    xl = pd.ExcelFile(path)
    dfs = []
    for sh in xl.sheet_names:
        d = xl.parse(sh)
        # Try to build a DatetimeIndex
        if not isinstance(d.index, pd.DatetimeIndex):
            for cand in ["Datetime","datetime","date","timestamp", d.columns[0]]:
                if cand in d.columns:
                    try:
                        d["__ts__"] = pd.to_datetime(d[cand])
                        d = d.drop(columns=[cand])
                        d.index = d["__ts__"]; d.drop(columns=["__ts__"], inplace=True)
                        break
                    except Exception:
                        pass
        d.index = pd.to_datetime(d.index).floor("h")
        d.sort_index(inplace=True)
        dfs.append(d)
    return dfs


# ------------------------- Artifact loader -------------------------
_ART_RE = re.compile(
    r"^(?P<lag>(?:LagDrop|lagdrop|nolags|no_lags|own(?:_lags)?|mutual(?:_lags)?|"
    r"Shared_mutual_lags|mi_top_k_nolags|SFS(?:_shared)?))?_?"
    r"(?P<model>LGBM|XGB|CatBoost|HGBR|GBR)_sheet(?P<sheet>\d+)_"
    r"(?P<target>P(?:_Power)?|Q(?:_Power)?)_best$",
    re.IGNORECASE
)

@dataclass
class Artifact:
    key: str
    base_path: Path
    model_name: str
    lag_policy: Optional[str]
    lag_token_raw: Optional[str]
    sheet: int
    target: str
    meta: Dict
    params: Dict
    features: List[str]
    recalc_features: List[str]
    model_file: Optional[Path] = None

def _norm_lag_policy(token: Optional[str]) -> Optional[str]:
    if not token: return None
    t = token.lower()
    if "lagdrop" in t or "nolags" in t or "no_lags" in t or "mi_top_k_nolags" in t:
        return "drop"
    if t == "own":
        return "own"
    if "mutual" in t:
        return "mutual"
    return None

def read_artifacts(models_dir: str) -> List[Artifact]:
    d = Path(models_dir)
    if not d.exists():
        raise FileNotFoundError(f"MODELS_DIR not found: {models_dir}")
    metas = list(d.glob("*_best.meta.json"))
    if not metas:
        raise RuntimeError(f"No '*_best.meta.json' found under {models_dir}")

    out: List[Artifact] = []
    for meta_path in metas:
        m = _ART_RE.match(meta_path.name.rsplit(".meta.json", 1)[0])
        print(m)
        if not m:
            warnings.warn(f"Skipping {meta_path.name}: filename pattern not recognized.")
            continue

        lag_token_raw = m.group("lag")
        lag_policy = _norm_lag_policy(lag_token_raw)
        model_name = m.group("model")
        sheet = int(m.group("sheet"))
        target = m.group("target")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        params = dict(meta.get("best_params") or {})
        features = list(meta.get("features") or [])
        recalc_features = list(meta.get("recalc_features") or [])
        if meta.get("lag_policy"):
            lag_policy = meta["lag_policy"]

        base_no_suffix = Path(str(meta_path).rsplit(".meta.json", 1)[0])
        model_file = None
        for ext in (".cbm",".json",".txt",".joblib"):
            cand = base_no_suffix.with_suffix(ext)
            if cand.exists():
                model_file = cand
                break

        key = f"{lag_policy or 'unknown'}|{model_name}|sheet{sheet}|{target}"
        out.append(Artifact(
            key=key, base_path=base_no_suffix, model_name=model_name, lag_policy=lag_policy,
            lag_token_raw=lag_token_raw, sheet=sheet, target=target, meta=meta, params=params,
            features=features, recalc_features=recalc_features, model_file=model_file
        ))
    return out

def recover_params_from_model_file(art: Artifact) -> Dict:
    """Recover params from native model file if meta lacks them."""
    if art.params:
        return art.params
    if art.model_file is None:
        return art.params
    suf = art.model_file.suffix.lower()
    try:
        if suf == ".json" and _pkg.get("XGB") is not None:
            xgb = _pkg["XGB"]
            tmp = xgb.XGBRegressor()
            tmp.load_model(str(art.model_file))
            p = tmp.get_params(deep=True)
            return p or art.params
        if suf == ".txt" and _pkg.get("LGBM") is not None:
            lgb = _pkg["LGBM"]
            booster = lgb.Booster(model_file=str(art.model_file))
            p = dict(booster.params or {})
            try:
                p.setdefault("n_estimators", booster.num_trees())
            except Exception:
                pass
            return p or art.params
    except Exception:
        pass
    return art.params


# ------------------------- Rolling predictors -------------------------
def roll_predict_joint(df: pd.DataFrame,
                       test_index: pd.DatetimeIndex,
                       tP: str, tQ: str,
                       estP, estQ,
                       allowedP: List[str], recalcP: List[str],
                       allowedQ: List[str], recalcQ: List[str],
                       modelP: str, modelQ: str,
                       catcolsP: List[str], catcolsQ: List[str],
                       trainedP: List[str], trainedQ: List[str],
                       catlvlsP: Dict[str, List[str]], catlvlsQ: Dict[str, List[str]],
                       oheP: List[str], oheQ: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Joint 1-step recursion for P & Q (for 'mutual' lag policy)."""
    yhist = {tP: df[tP].copy(), tQ: df[tQ].copy()}
    predsP, predsQ = [], []
    for ts in test_index:
        baseP = df.loc[[ts], allowedP].copy()
        for c in recalcP:
            baseP.loc[ts, c] = recompute_feature_at(ts, c, yhist)
        XP, _, _, _ = prepare_X_for_model(modelP, baseP, catcolsP, xgb_native_categorical=XGB_NATIVE_CATS)
        XP = freeze_categories(XP, catlvlsP)
        XP = align_to_columns(XP, trainedP)
        if modelP in {"HGBR","GBR"} or (modelP == "XGB" and not XGB_NATIVE_CATS):
            if oheP: XP[oheP] = XP[oheP].fillna(0.0)
        XP = safe_for_model(XP, modelP)
        yhatP = float(estP.predict(XP)[0])

        baseQ = df.loc[[ts], allowedQ].copy()
        for c in recalcQ:
            baseQ.loc[ts, c] = recompute_feature_at(ts, c, yhist)
        XQ, _, _, _ = prepare_X_for_model(modelQ, baseQ, catcolsQ, xgb_native_categorical=XGB_NATIVE_CATS)
        XQ = freeze_categories(XQ, catlvlsQ)
        XQ = align_to_columns(XQ, trainedQ)
        if modelQ in {"HGBR","GBR"} or (modelQ == "XGB" and not XGB_NATIVE_CATS):
            if oheQ: XQ[oheQ] = XQ[oheQ].fillna(0.0)
        XQ = safe_for_model(XQ, modelQ)
        yhatQ = float(estQ.predict(XQ)[0])

        predsP.append(yhatP); predsQ.append(yhatQ)
        yhist[tP].loc[ts] = yhatP
        yhist[tQ].loc[ts] = yhatQ

    return np.asarray(predsP, dtype=float), np.asarray(predsQ, dtype=float)

def roll_predict_own_single(df: pd.DataFrame,
                            test_index: pd.DatetimeIndex,
                            tgt: str,
                            est,
                            allowed: List[str], recalc: List[str],
                            model_name: str,
                            catcols: List[str],
                            trained_cols: List[str],
                            catlvls: Dict[str, List[str]],
                            ohe_cols: List[str]) -> np.ndarray:
    """Single-target 1-step recursion for 'own' lag policy."""
    yhist = {tgt: df[tgt].copy()}
    preds = []
    for ts in test_index:
        base = df.loc[[ts], allowed].copy()
        for c in recalc:
            base.loc[ts, c] = recompute_feature_at(ts, c, yhist)
        X, _, _, _ = prepare_X_for_model(model_name, base, catcols, xgb_native_categorical=XGB_NATIVE_CATS)
        X = freeze_categories(X, catlvls)
        X = align_to_columns(X, trained_cols)
        if model_name in {"HGBR","GBR"} or (model_name == "XGB" and not XGB_NATIVE_CATS):
            if ohe_cols: X[ohe_cols] = X[ohe_cols].fillna(0.0)
        X = safe_for_model(X, model_name)
        yhat = float(est.predict(X)[0])
        preds.append(yhat)
        yhist[tgt].loc[ts] = yhat
    return np.asarray(preds, dtype=float)

def direct_predict(df: pd.DataFrame,
                   test_index: pd.DatetimeIndex,
                   est, allowed: List[str],
                   model_name: str, catcols: List[str],
                   trained_cols: List[str], catlvls: Dict[str, List[str]],
                   ohe_cols: List[str]) -> np.ndarray:
    """No recursion: build full test matrix and predict at once."""
    base = df.loc[test_index, allowed].copy()
    X, _, _, _ = prepare_X_for_model(model_name, base, catcols, xgb_native_categorical=XGB_NATIVE_CATS)
    X = freeze_categories(X, catlvls)
    X = align_to_columns(X, trained_cols)
    if model_name in {"HGBR","GBR"} or (model_name == "XGB" and not XGB_NATIVE_CATS):
        if ohe_cols: X[ohe_cols] = X[ohe_cols].fillna(0.0)
    X = safe_for_model(X, model_name)
    return np.asarray(est.predict(X), dtype=float).ravel()


# ------------------------- PREDICTION INTERVALS / SIMULATIONS -------------------------
def moving_block_bootstrap_indices(T: int, block_size: int, B: int) -> List[np.ndarray]:
    if block_size < 1: block_size = 1
    num_blocks = int(math.ceil(T / block_size))
    out = []
    for _ in range(B):
        starts = np.random.randint(0, T, size=num_blocks)
        idx = []
        for s in starts:
            block = [(s + k) % T for k in range(block_size)]
            idx.extend(block)
        out.append(np.array(idx[:T], dtype=int))
    return out

def residual_block_pi(
    y_train: np.ndarray,
    yhat_train: np.ndarray,
    yhat_test: np.ndarray,
    *,
    block_size: int = 24,
    R: int = 1000,
    levels: tuple[float, ...] = (0.80, 0.95),
    match_hour: bool = True,
    idx_train: pd.DatetimeIndex | None = None,
    idx_test: pd.DatetimeIndex | None = None
) -> tuple[dict[float, tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """
    Residual resampling to form prediction intervals.
    Returns (intervals_dict, sims) with sims shape (R, T).
    """
    y_train = np.asarray(y_train, dtype=float).ravel()
    yhat_train = np.asarray(yhat_train, dtype=float).ravel()
    yhat_test = np.asarray(yhat_test, dtype=float).ravel()
    Tte = yhat_test.size

    if match_hour and (idx_train is not None) and (idx_test is not None):
        res = y_train - yhat_train
        by_hour = {h: [] for h in range(24)}
        for r, ts in zip(res, idx_train):
            h = int(pd.Timestamp(ts).hour)
            if np.isfinite(r):
                by_hour[h].append(float(r))
        for h in range(24):
            if not by_hour[h]:
                by_hour[h] = [0.0]
        sims = np.empty((R, Tte), dtype=float)
        for j, ts in enumerate(idx_test):
            h = int(pd.Timestamp(ts).hour)
            pool = np.asarray(by_hour[h], dtype=float)
            draw = np.random.choice(pool, size=R, replace=True)
            sims[:, j] = yhat_test[j] + draw
    else:
        res = (y_train - yhat_train)
        res = res[np.isfinite(res)]
        if res.size == 0:
            res = np.array([0.0], dtype=float)
        idxs = moving_block_bootstrap_indices(len(res), block_size, R)
        sims = np.empty((R, Tte), dtype=float)
        for b in range(R):
            rr = res[idxs[b]]
            if rr.size < Tte:
                rr = np.resize(rr, Tte)
            sims[b, :] = yhat_test + rr[:Tte]

    out: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for lev in levels:
        qlo = (1.0 - lev) / 2.0
        qhi = 1.0 - qlo
        L = np.quantile(sims, qlo, axis=0)
        U = np.quantile(sims, qhi, axis=0)
        out[float(lev)] = (L, U)
    return out, sims


# ------------------------- Experiment on a sheet (single pass) -------------------------
@dataclass
class ExperimentResult:
    exp: str
    model_name: str
    lag_policy: str
    target: str
    other_target: str
    y_true: np.ndarray
    y_pred: np.ndarray
    index: pd.DatetimeIndex
    metrics: Dict[str, float]
    pred_df: Optional[pd.DataFrame] = None         # per-timestamp predictions (+ intervals/quantiles/scores)
    daily_metrics: Optional[pd.DataFrame] = None   # per-day metrics (one row per day)

def _add_quantiles_intervals_scores_to_df(
    base_df: pd.DataFrame,
    sims: np.ndarray,
    y_true_vec: np.ndarray,
    *,
    add_all_intervals: bool = True
) -> pd.DataFrame:
    """
    Given a base df with columns ts,date,y_true,y_pred and simulated draws (R,T),
    append:
      - quantiles Q05..Q95
      - equal-tailed intervals Lxx/Uxx for coverages in ALL_INTERVAL_COVERAGES (0.05..0.95)
      - CRPS (per forecast)
      - Pinball_0.xx per tau in QUANTILE_GRID
    """
    out = base_df.copy()
    T = out.shape[0]

    # Quantiles for tau grid
    if QUANTILE_GRID:
        qmat = np.quantile(sims, QUANTILE_GRID, axis=0)  # shape (len(grid), T)
        for tau, row in zip(QUANTILE_GRID, qmat):
            name = f"Q{int(round(100*tau)):02d}"
            out[name] = row.astype(float)

    # Full coverage interval grid (includes 80/95 as part of it)
    if add_all_intervals and ALL_INTERVAL_COVERAGES:
        for cov in ALL_INTERVAL_COVERAGES:
            qlo = (1.0 - cov) / 2.0
            qhi = 1.0 - qlo
            L = np.quantile(sims, qlo, axis=0)
            U = np.quantile(sims, qhi, axis=0)
            out[f"L{int(round(100*cov)):02d}"] = L.astype(float)
            out[f"U{int(round(100*cov)):02d}"] = U.astype(float)

    # CRPS per forecast
    crps = crps_from_sims(y_true_vec, sims)  # (T,)
    out["CRPS"] = crps.astype(float)

    # Pinball per tau per forecast
    if QUANTILE_GRID:
        for tau in QUANTILE_GRID:
            q_col = f"Q{int(round(100*tau)):02d}"
            pl = pinball_loss(out["y_true"].values, out[q_col].values, tau)
            out[f"Pinball_{tau:.2f}"] = pl.astype(float)

    return out

def fit_refit_and_eval_sheet(df: pd.DataFrame,
                             split: Dict[str,str],
                             artP: Artifact, artQ: Artifact,
                             sheet_idx: int) -> List[ExperimentResult]:
    np.random.seed(RANDOM_SEED)
    tP, tQ = identify_targets(df)

    train_mask = (df.index >= pd.Timestamp(split["train_start"])) & (df.index <= pd.Timestamp(split["train_end"]))
    test_mask  = (df.index >= pd.Timestamp(split["test_start"]))  & (df.index <= pd.Timestamp(split["test_end"]))
    if not train_mask.any() or not test_mask.any():
        warnings.warn("Empty train or test mask after clipping. Check your dates.")
        return []

    tr_start, tr_end = df.index[train_mask][0], df.index[train_mask][-1]
    te_start, te_end = df.index[test_mask][0], df.index[test_mask][-1]
    print(f"  > TRAIN {tr_start} .. {tr_end}  (n={train_mask.sum()}) | TEST {te_start} .. {te_end} (n={test_mask.sum()})")

    # Build estimators
    estP = build_estimator_from_params(artP.model_name, artP.params, use_gpu=True)
    estQ = build_estimator_from_params(artQ.model_name, artQ.params, use_gpu=True)

    # Features present in this sheet
    allowedP = [c for c in artP.features if c in df.columns]
    allowedQ = [c for c in artQ.features if c in df.columns]
    recalcP  = [c for c in artP.recalc_features if c in allowedP]
    recalcQ  = [c for c in artQ.recalc_features if c in allowedQ]
    if not allowedP or not allowedQ:
        warnings.warn("Some artifact has no overlapping features in this sheet; skipping.")
        return []

    print(f"    [{artP.model_name}|{artP.lag_policy or 'drop'}|PQsheet {sheet_idx}] "
          f"P: features={len(allowedP)} (recalc={len(recalcP)}) | "
          f"Q: features={len(allowedQ)} (recalc={len(recalcQ)})")

    # Train matrices (no recompute in TRAIN)
    XtrP = df.loc[train_mask, allowedP].copy()
    XtrQ = df.loc[train_mask, allowedQ].copy()

    _, catsP = classify_features(XtrP, KNOWN_CATS)
    _, catsQ = classify_features(XtrQ, KNOWN_CATS)

    XtrP, trainedP, catidxP, oheP = prepare_X_for_model(artP.model_name, XtrP, catsP, xgb_native_categorical=XGB_NATIVE_CATS)
    XtrQ, trainedQ, catidxQ, oheQ = prepare_X_for_model(artQ.model_name, XtrQ, catsQ, xgb_native_categorical=XGB_NATIVE_CATS)

    # Save cat levels for predict time
    catlvlsP = {c: list(XtrP[c].cat.categories) for c in XtrP.columns if isinstance(XtrP[c].dtype, CategoricalDtype)}
    catlvlsQ = {c: list(XtrQ[c].cat.categories) for c in XtrQ.columns if isinstance(XtrQ[c].dtype, CategoricalDtype)}

    if artP.model_name.upper() in {"HGBR","GBR"} or (artP.model_name.upper()=="XGB" and not XGB_NATIVE_CATS):
        if oheP: XtrP[oheP] = XtrP[oheP].fillna(0.0)
    if artQ.model_name.upper() in {"HGBR","GBR"} or (artQ.model_name.upper()=="XGB" and not XGB_NATIVE_CATS):
        if oheQ: XtrQ[oheQ] = XtrQ[oheQ].fillna(0.0)

    XtrP = safe_for_model(XtrP, artP.model_name)
    XtrQ = safe_for_model(XtrQ, artQ.model_name)

    ytrP = df.loc[train_mask, tP].values
    ytrQ = df.loc[train_mask, tQ].values

    # Fit
    if artP.model_name.upper() == "CATBOOST" and _pkg["CatBoost"] is not None:
        estP.fit(align_to_columns(XtrP, trainedP), ytrP, cat_features=catidxP, verbose=False)
    else:
        estP.fit(align_to_columns(XtrP, trainedP), ytrP)
    if artQ.model_name.upper() == "CATBOOST" and _pkg["CatBoost"] is not None:
        estQ.fit(align_to_columns(XtrQ, trainedQ), ytrQ, cat_features=catidxQ, verbose=False)
    else:
        estQ.fit(align_to_columns(XtrQ, trainedQ), ytrQ)

    # In-sample preds for PI residuals
    yhat_tr_P = np.asarray(estP.predict(align_to_columns(XtrP, trainedP)), dtype=float).ravel()
    yhat_tr_Q = np.asarray(estQ.predict(align_to_columns(XtrQ, trainedQ)), dtype=float).ravel()

    te_index = df.index[test_mask]
    lp = (artP.lag_policy or artQ.lag_policy or "drop").lower()

    if lp == "mutual":
        yhat_te_P, yhat_te_Q = roll_predict_joint(
            df=df, test_index=te_index,
            tP=tP, tQ=tQ,
            estP=estP, estQ=estQ,
            allowedP=allowedP, recalcP=recalcP,
            allowedQ=allowedQ, recalcQ=recalcQ,
            modelP=artP.model_name.upper(), modelQ=artQ.model_name.upper(),
            catcolsP=catsP, catcolsQ=catsQ,
            trainedP=trainedP, trainedQ=trainedQ,
            catlvlsP=catlvlsP, catlvlsQ=catlvlsQ,
            oheP=oheP, oheQ=oheQ
        )
    elif lp == "own":
        yhat_te_P = roll_predict_own_single(
            df=df, test_index=te_index, tgt=tP, est=estP,
            allowed=allowedP, recalc=recalcP,
            model_name=artP.model_name.upper(),
            catcols=catsP, trained_cols=trainedP,
            catlvls=catlvlsP, ohe_cols=oheP
        )
        yhat_te_Q = roll_predict_own_single(
            df=df, test_index=te_index, tgt=tQ, est=estQ,
            allowed=allowedQ, recalc=recalcQ,
            model_name=artQ.model_name.upper(),
            catcols=catsQ, trained_cols=trainedQ,
            catlvls=catlvlsQ, ohe_cols=oheQ
        )
    else:
        yhat_te_P = direct_predict(df, te_index, estP, allowedP, artP.model_name.upper(), catsP, trainedP, catlvlsP, oheP)
        yhat_te_Q = direct_predict(df, te_index, estQ, allowedQ, artQ.model_name.upper(), catsQ, trainedQ, catlvlsQ, oheQ)

    # Metrics
    y_true_P = df.loc[test_mask, tP].values
    y_true_Q = df.loc[test_mask, tQ].values

    results: List[ExperimentResult] = []
    for (tag_model, lag_pol, tgt, other, y_true, yhat_te, yhat_tr, ytr) in [
        (artP.model_name, artP.lag_policy or "drop", tP, tQ, y_true_P, yhat_te_P, yhat_tr_P, ytrP),
        (artQ.model_name, artQ.lag_policy or "drop", tQ, tP, y_true_Q, yhat_te_Q, yhat_tr_Q, ytrQ),
    ]:
        ms = evaluate_predictions(y_true, yhat_te, y_train=ytr)

        # Simulations + intervals & scores
        if BUILD_INTERVALS:
            # include both main interval levels and the full coverage grid
            cov_for_engine = tuple(sorted(set(INTERVAL_LEVELS) | set(ALL_INTERVAL_COVERAGES)))
            pis, sims = residual_block_pi(
                y_train=ytr, yhat_train=yhat_tr, yhat_test=yhat_te,
                block_size=BLOCK_SIZE_PI, R=N_SIM_PI, levels=cov_for_engine,
                match_hour=True, idx_train=df.index[train_mask], idx_test=te_index
            )
        else:
            pis, sims = {}, np.empty((0, len(yhat_te)), dtype=float)

        # Base per-timestamp frame
        pred_df = pd.DataFrame({
            "ts": te_index, "date": te_index.date,
            "y_true": y_true, "y_pred": yhat_te,
        })
        # Add 80/95 from pis (kept for compatibility)
        if pis:
            if 0.80 in pis:
                pred_df["L80"], pred_df["U80"] = pis[0.80][0], pis[0.80][1]
            if 0.95 in pis:
                pred_df["L95"], pred_df["U95"] = pis[0.95][0], pis[0.95][1]

        # Add full quantiles / full interval grid / CRPS / Pinball
        if sims.size:
            pred_df = _add_quantiles_intervals_scores_to_df(pred_df, sims, y_true, add_all_intervals=True)

        # Per-day metrics (+ new scores)
        def _day_metrics(d: pd.DataFrame) -> pd.Series:
            out = {
                "n": len(d),
                "RMSE": rmse(d["y_true"], d["y_pred"]),
                "MAE":  mae(d["y_true"], d["y_pred"]),
                "MAPE%": mape(d["y_true"], d["y_pred"]),
                "SMAPE%": smape(d["y_true"], d["y_pred"]),
                "WMAPE%": wmape(d["y_true"], d["y_pred"]),
                "MASE": mase(d["y_true"].values, d["y_pred"].values, ytr),
            }
            if "CRPS" in d.columns:
                out["CRPS"] = float(np.mean(d["CRPS"].values))
            # mean pinball across the tau grid (one column per tau)
            for tau in QUANTILE_GRID:
                col = f"Pinball_{tau:.2f}"
                if col in d.columns:
                    out[col] = float(np.mean(d[col].values))
            return pd.Series(out)

        day_df = pred_df.groupby("date", observed=True).apply(_day_metrics)
        day_df.index.name = "date"

        tag = f"{tag_model}|{lag_pol}|sheet{sheet_idx}|{tgt}"
        results.append(ExperimentResult(
            exp=tag, model_name=tag_model, lag_policy=lag_pol, target=tgt, other_target=other,
            y_true=y_true, y_pred=yhat_te, index=te_index,
            metrics=ms, pred_df=pred_df, daily_metrics=day_df
        ))
    return results


def fit_refit_and_eval_sheet_walkforward(df: pd.DataFrame,
                                         artP: Artifact, artQ: Artifact,
                                         sheet_idx: int,
                                         *,
                                         train_start: Optional[str],
                                         test_start: str,
                                         test_end: str) -> List[ExperimentResult]:
    np.random.seed(RANDOM_SEED)
    tP, tQ = identify_targets(df)

    allowedP = [c for c in artP.features if c in df.columns]
    allowedQ = [c for c in artQ.features if c in df.columns]
    recalcP  = [c for c in artP.recalc_features if c in allowedP]
    recalcQ  = [c for c in artQ.recalc_features if c in allowedQ]
    if not allowedP or not allowedQ:
        warnings.warn("Artifact has no overlapping features with this PQ sheet.")
        return []

    _, catsP_all = classify_features(df[allowedP], KNOWN_CATS)
    _, catsQ_all = classify_features(df[allowedQ], KNOWN_CATS)

    lp = (artP.lag_policy or artQ.lag_policy or "drop").lower()

    # we will accumulate per-target prediction dataframes and daily rows
    acc = {tP: {"dfs": [], "daily": []}, tQ: {"dfs": [], "daily": []}}

    for block in iter_day_blocks_for_sheet(df.index,
                                           train_start=train_start,
                                           test_start=test_start,
                                           test_end=test_end,
                                           horizon_hours=DAY_AHEAD_HOURS):
        tr_mask = (df.index >= pd.Timestamp(block["train_start"])) & (df.index <= pd.Timestamp(block["train_end"]))
        te_index = pd.date_range(block["test_start"], block["test_end"], freq="h")
        te_index = te_index[te_index.isin(df.index)]
        if tr_mask.sum() <= 24 or len(te_index) == 0:
            continue

        # fresh estimators
        estP = build_estimator_from_params(artP.model_name, artP.params, use_gpu=True)
        estQ = build_estimator_from_params(artQ.model_name, artQ.params, use_gpu=True)

        XtrP = df.loc[tr_mask, allowedP].copy()
        XtrQ = df.loc[tr_mask, allowedQ].copy()

        XtrP, trainedP, catidxP, oheP = prepare_X_for_model(artP.model_name, XtrP, catsP_all, xgb_native_categorical=XGB_NATIVE_CATS)
        XtrQ, trainedQ, catidxQ, oheQ = prepare_X_for_model(artQ.model_name, XtrQ, catsQ_all, xgb_native_categorical=XGB_NATIVE_CATS)

        catlvlsP = {c: list(XtrP[c].cat.categories) for c in XtrP.columns if isinstance(XtrP[c].dtype, CategoricalDtype)}
        catlvlsQ = {c: list(XtrQ[c].cat.categories) for c in XtrQ.columns if isinstance(XtrQ[c].dtype, CategoricalDtype)}

        if artP.model_name.upper() in {"HGBR","GBR"} or (artP.model_name.upper()=="XGB" and not XGB_NATIVE_CATS):
            if oheP: XtrP[oheP] = XtrP[oheP].fillna(0.0)
        if artQ.model_name.upper() in {"HGBR","GBR"} or (artQ.model_name.upper()=="XGB" and not XGB_NATIVE_CATS):
            if oheQ: XtrQ[oheQ] = XtrQ[oheQ].fillna(0.0)
        XtrP = safe_for_model(XtrP, artP.model_name)
        XtrQ = safe_for_model(XtrQ, artQ.model_name)

        ytrP = df.loc[tr_mask, tP].values
        ytrQ = df.loc[tr_mask, tQ].values

        if artP.model_name.upper() == "CATBOOST" and _pkg["CatBoost"] is not None:
            estP.fit(align_to_columns(XtrP, trainedP), ytrP, cat_features=catidxP, verbose=False)
        else:
            estP.fit(align_to_columns(XtrP, trainedP), ytrP)
        if artQ.model_name.upper() == "CATBOOST" and _pkg["CatBoost"] is not None:
            estQ.fit(align_to_columns(XtrQ, trainedQ), ytrQ, cat_features=catidxQ, verbose=False)
        else:
            estQ.fit(align_to_columns(XtrQ, trainedQ), ytrQ)

        # in-sample residuals for intervals
        yhat_tr_P = np.asarray(estP.predict(align_to_columns(XtrP, trainedP)), dtype=float).ravel()
        yhat_tr_Q = np.asarray(estQ.predict(align_to_columns(XtrQ, trainedQ)), dtype=float).ravel()

        # predict this day
        if lp == "mutual":
            yhatP, yhatQ = roll_predict_joint(
                df=df, test_index=te_index,
                tP=tP, tQ=tQ,
                estP=estP, estQ=estQ,
                allowedP=allowedP, recalcP=recalcP,
                allowedQ=allowedQ, recalcQ=recalcQ,
                modelP=artP.model_name.upper(), modelQ=artQ.model_name.upper(),
                catcolsP=catsP_all, catcolsQ=catsQ_all,
                trainedP=trainedP, trainedQ=trainedQ,
                catlvlsP=catlvlsP, catlvlsQ=catlvlsQ,
                oheP=oheP, oheQ=oheQ
            )
        elif lp == "own":
            yhatP = roll_predict_own_single(df, te_index, tP, estP, allowedP, recalcP,
                                            artP.model_name.upper(), catsP_all, trainedP, catlvlsP, oheP)
            yhatQ = roll_predict_own_single(df, te_index, tQ, estQ, allowedQ, recalcQ,
                                            artQ.model_name.upper(), catsQ_all, trainedQ, catlvlsQ, oheQ)
        else:
            yhatP = direct_predict(df, te_index, estP, allowedP, artP.model_name.upper(), catsP_all, trainedP, catlvlsP, oheP)
            yhatQ = direct_predict(df, te_index, estQ, allowedQ, artQ.model_name.upper(), catsQ_all, trainedQ, catlvlsQ, oheQ)

        # true values for the day
        ytP = df.loc[te_index, tP].values
        ytQ = df.loc[te_index, tQ].values

        # day-level simulations
        if BUILD_INTERVALS:
            cov_for_engine = tuple(sorted(set(INTERVAL_LEVELS) | set(ALL_INTERVAL_COVERAGES)))
            pisP, simsP = residual_block_pi(ytrP, yhat_tr_P, yhatP,
                                            block_size=BLOCK_SIZE_PI, R=N_SIM_PI,
                                            levels=cov_for_engine,
                                            match_hour=True, idx_train=df.index[tr_mask], idx_test=te_index)
            pisQ, simsQ = residual_block_pi(ytrQ, yhat_tr_Q, yhatQ,
                                            block_size=BLOCK_SIZE_PI, R=N_SIM_PI,
                                            levels=cov_for_engine,
                                            match_hour=True, idx_train=df.index[tr_mask], idx_test=te_index)
        else:
            pisP, simsP = {}, np.empty((0, len(yhatP)))
            pisQ, simsQ = {}, np.empty((0, len(yhatQ)))

        for tgt, yt, yhat, pis, sims, acc_slot in [
            (tP, ytP, yhatP, pisP, simsP, acc[tP]),
            (tQ, ytQ, yhatQ, pisQ, simsQ, acc[tQ]),
        ]:
            # base frame for this day
            df_day = pd.DataFrame({
                "ts": te_index, "date": te_index.date,
                "y_true": yt, "y_pred": yhat
            })
            # keep L80/U80 and L95/U95 for compatibility
            if 0.80 in pis:
                df_day["L80"], df_day["U80"] = pis[0.80][0], pis[0.80][1]
            if 0.95 in pis:
                df_day["L95"], df_day["U95"] = pis[0.95][0], pis[0.95][1]

            if sims.size:
                df_day = _add_quantiles_intervals_scores_to_df(df_day, sims, yt, add_all_intervals=True)

            # per-day row with averages
            row = {
                "date": block["date"],
                "n": len(yt),
                "RMSE": rmse(yt, yhat),
                "MAE":  mae(yt, yhat),
                "MAPE%": mape(yt, yhat),
                "SMAPE%": smape(yt, yhat),
                "WMAPE%": wmape(yt, yhat),
                "MASE": mase(yt, yhat, (ytrP if tgt == tP else ytrQ)),
            }
            if "CRPS" in df_day.columns:
                row["CRPS"] = float(np.mean(df_day["CRPS"].values))
            for tau in QUANTILE_GRID:
                col = f"Pinball_{tau:.2f}"
                if col in df_day.columns:
                    row[col] = float(np.mean(df_day[col].values))

            acc_slot["dfs"].append(df_day)
            acc_slot["daily"].append(row)

    # build ExperimentResult objects (overall across all days)
    results: List[ExperimentResult] = []
    for (art, tgt, other, slot) in [
        (artP, tP, tQ, acc[tP]),
        (artQ, tQ, tP, acc[tQ]),
    ]:
        if not slot["dfs"]:
            continue
        pred_df = pd.concat(slot["dfs"], ignore_index=True)
        pred_df.sort_values("ts", inplace=True)
        y_true = pred_df["y_true"].values
        y_pred = pred_df["y_pred"].values
        index  = pd.to_datetime(pred_df["ts"].values)
        y_train_full = df.loc[df.index < index.min(), tgt].values
        ms  = evaluate_predictions(y_true, y_pred, y_train=y_train_full)

        daily_df = (pd.DataFrame(slot["daily"])
                      .set_index("date")
                      .sort_index())

        tag = f"{art.model_name}|{art.lag_policy or 'drop'}|sheet{sheet_idx}|{tgt}"
        results.append(ExperimentResult(
            exp=tag, model_name=art.model_name, lag_policy=art.lag_policy or "drop",
            target=tgt, other_target=other,
            y_true=y_true, y_pred=y_pred, index=index,
            metrics=ms, pred_df=pred_df, daily_metrics=daily_df
        ))
    return results


# ------------------------- Grouping & selection -------------------------
def group_artifacts_by_family(arts: List[Artifact]) -> Dict[Tuple[str,str], Dict[int, Dict[str, Artifact]]]:
    groups: Dict[Tuple[str,str], Dict[int, Dict[str, Artifact]]] = {}
    for a in arts:
        fam = (a.lag_token_raw or (a.lag_policy or "unknown")).strip()
        key = (fam, a.model_name.upper())
        groups.setdefault(key, {})
        groups[key].setdefault(a.sheet, {})
        groups[key][a.sheet][a.target] = a
    return groups

def choose_best_pq_sheet_for_pair(
    artP: Artifact, artQ: Artifact, PQ_list: List[pd.DataFrame],
    *, train_start, train_end, test_start, test_end
) -> Optional[int]:
    reqP = set(artP.features or [])
    reqQ = set(artQ.features or [])
    candidates: List[Tuple[int,int,int]] = []
    for sheet_idx, df in enumerate(PQ_list):
        cols = set(df.columns)
        if (reqP.issubset(cols) and reqQ.issubset(cols)):
            try:
                split = build_split_for_sheet(df.index, train_start, train_end, test_start, test_end)
                mask = (df.index >= pd.Timestamp(split["train_start"])) & (df.index <= pd.Timestamp(split["train_end"]))
                train_rows = int(mask.sum())
            except Exception:
                train_rows = 0
            if train_rows > 0:
                candidates.append((train_rows, -sheet_idx, sheet_idx))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]

def run_experiment_for_pair(
    family_key: Tuple[str, str],
    artP: Artifact,
    artQ: Artifact,
    PQ_list: List[pd.DataFrame]
) -> Tuple[Dict[str, Dict[str, ExperimentResult]], Optional[str]]:
    fam_token, model_name = family_key
    artP.params = recover_params_from_model_file(artP) or artP.params
    artQ.params = recover_params_from_model_file(artQ) or artQ.params

    best_sheet = choose_best_pq_sheet_for_pair(
        artP, artQ, PQ_list,
        train_start=TRAIN_START, train_end=TRAIN_END,
        test_start=TEST_START, test_end=TEST_END
    )
    if best_sheet is None:
        print(f"  (skip) No PQ sheet covers all features for this pair: {fam_token}|{model_name}|P={artP.base_path.name} | Q={artQ.base_path.name}")
        return {}, None

    df = PQ_list[best_sheet]
    split = build_split_for_sheet(df.index, TRAIN_START, TRAIN_END, TEST_START, TEST_END)
    print(f"\n=== Running instance: FAMILY={fam_token}  MODEL={model_name}  → PQ sheet {best_sheet}")
    print(f"    using P={artP.base_path.name}")
    print(f"    using Q={artQ.base_path.name}")
    print(f"    split: {split}")

    lag_token_raw = artP.lag_token_raw or artQ.lag_token_raw or None
    pair_sig = short_sig(artP.base_path.name, artQ.base_path.name,
                         str(len(artP.features)), str(len(artQ.features)),
                         str(len(artP.recalc_features)), str(len(artQ.recalc_features)))
    if lag_token_raw:
        excel_stub = f"{lag_token_raw}_{model_name}_sheet{best_sheet}_{pair_sig}_PQ_Power_best"
    else:
        excel_stub = f"{model_name}_sheet{best_sheet}_{pair_sig}_PQ_Power_best"

    all_results: Dict[str, Dict[str, ExperimentResult]] = {}
    if WALKFORWARD_DAILY:
        res_sheet = fit_refit_and_eval_sheet_walkforward(
            df, artP, artQ, sheet_idx=best_sheet,
            train_start=TRAIN_START, test_start=TEST_START, test_end=TEST_END
        )
    else:
        res_sheet = fit_refit_and_eval_sheet(
            df, split, artP, artQ, sheet_idx=best_sheet
        )
    for r in res_sheet:
        all_results.setdefault(r.target, {})
        all_results[r.target][r.exp] = r

    return all_results, excel_stub


def save_results_for_family(
    family_key: Tuple[str,str],
    all_results: Dict[str, Dict[str, ExperimentResult]],
    *,
    train_start, train_end, test_start, test_end,
    excel_stub: Optional[str] = None,
) -> None:
    """
    Writer: emit only the four sheets requested:
      - P_Power_predictions
      - P_Power_daily_metrics
      - Q_Power_predictions
      - Q_Power_daily_metrics
    """
    if not all_results:
        print("No results to save for this experiment.")
        return

    fam_token, model_name = family_key
    tgt_keys = list(all_results.keys())

    def _canon(t: str) -> str:
        t = str(t)
        if t in ("P", "P_Power"): return "P_Power"
        if t in ("Q", "Q_Power"): return "Q_Power"
        return t

    if excel_stub:
        out_path = SAVE_DIR / f"{excel_stub}.xlsx"
    else:
        def _tag(x):
            return ("auto" if x in (None, "auto") else pd.Timestamp(x).strftime("%Y%m%d%H"))
        rtag = f"{_tag(train_start)}_{_tag(train_end)}__{_tag(test_start)}_{_tag(test_end)}"
        out_path = SAVE_DIR / f"pq_eval_{fam_token}_{model_name}_{rtag}.xlsx"

    engine = choose_excel_engine()
    if engine is not None:
        out_path = ensure_unique_path(out_path)
        print(f"Writing Excel with engine='{engine}' → {out_path}")

        with pd.ExcelWriter(out_path, engine=engine) as xl:
            for tk in tgt_keys:
                tk_can = _canon(tk)

                # ---- predictions (stack all experiments for this target) ----
                preds_list = []
                for tag, r in all_results[tk].items():
                    if r.pred_df is not None and not r.pred_df.empty:
                        dfp = r.pred_df.copy()
                        dfp.insert(0, "exp", tag)
                        preds_list.append(dfp)
                if preds_list:
                    preds = pd.concat(preds_list, ignore_index=True)
                    preds.to_excel(
                        xl, sheet_name=_safe_sheet_name(f"{tk_can}_predictions"), index=False
                    )

                # ---- daily metrics (stack all experiments for this target) ----
                daily_list = []
                for tag, r in all_results[tk].items():
                    if r.daily_metrics is not None and not r.daily_metrics.empty:
                        dm = r.daily_metrics.copy()
                        dm.insert(0, "exp", tag)
                        daily_list.append(dm.reset_index().rename(columns={"index": "date"}))
                if daily_list:
                    daily = pd.concat(daily_list, ignore_index=True)
                    daily.to_excel(
                        xl, sheet_name=_safe_sheet_name(f"{tk_can}_daily_metrics"), index=False
                    )

        print(f"Saved results to: {out_path.resolve()}")

    else:
        csv_dir = SAVE_DIR / (excel_stub if excel_stub else f"pq_eval_{fam_token}_{model_name}_csv")
        csv_dir.mkdir(parents=True, exist_ok=True)
        print("No Excel writer engine installed. Writing minimal CSVs instead.")

        for tk in tgt_keys:
            tk_can = _canon(tk)
            preds_list = []
            for tag, r in all_results[tk].items():
                if r.pred_df is not None and not r.pred_df.empty:
                    dfp = r.pred_df.copy()
                    dfp.insert(0, "exp", tag)
                    preds_list.append(dfp)
            if preds_list:
                pd.concat(preds_list, ignore_index=True).to_csv(
                    csv_dir / f"{tk_can}_predictions.csv", index=False
                )

            daily_list = []
            for tag, r in all_results[tk].items():
                if r.daily_metrics is not None and not r.daily_metrics.empty:
                    dm = r.daily_metrics.copy()
                    dm.insert(0, "exp", tag)
                    daily_list.append(dm.reset_index().rename(columns={"index": "date"}))
            if daily_list:
                pd.concat(daily_list, ignore_index=True).to_csv(
                    csv_dir / f"{tk_can}_daily_metrics.csv", index=False
                )

        print(f"CSV results saved under: {csv_dir.resolve()}")
        print("To enable Excel output, install an engine, e.g.: 'pip install xlsxwriter' or 'pip install openpyxl'.")


# ------------------------- Main runner -------------------------
def main():
    np.random.seed(RANDOM_SEED)
    resolve_paths()

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"DATA_PATH not found: {DATA_PATH}")
    PQ_list = read_pq_xlsx(DATA_PATH)

    arts = read_artifacts(MODELS_DIR)
    if not arts:
        raise RuntimeError("No artifacts found.")

    for a in arts:
        a.params = recover_params_from_model_file(a) or a.params

    families = group_artifacts_by_family(arts)
    if not families:
        raise RuntimeError("No valid artifacts grouped by (family_token, model).")

    for family_key, sheets_map in families.items():
        usable: Dict[int, Dict[str, Artifact]] = {
            s: m for s, m in sheets_map.items()
            if ("P_Power" in m or "P" in m) and ("Q_Power" in m or "Q" in m)
        }
        if not usable:
            print(f"Skipping family {family_key}: no sheet has both targets.")
            continue

        for sheet_id, pair in sorted(usable.items()):
            artP = pair.get("P_Power", pair.get("P"))
            artQ = pair.get("Q_Power", pair.get("Q"))
            if not artP or not artQ:
                continue

            all_results, excel_stub = run_experiment_for_pair(family_key, artP, artQ, PQ_list)

            save_results_for_family(
                family_key, all_results,
                train_start=TRAIN_START, train_end=TRAIN_END,
                test_start=TEST_START, test_end=TEST_END,
                excel_stub=excel_stub,
            )

if __name__ == "__main__":
    main()

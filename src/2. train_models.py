# -*- coding: utf-8 -*-
from __future__ import annotations

import json, re, copy, warnings, math, os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Core learners
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression, SequentialFeatureSelector
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer

try:
    from lightgbm import LGBMRegressor
    import lightgbm as lgb
except Exception:
    LGBMRegressor = None
    lgb = None

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

import joblib
import optuna

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---- Regex for lag/rolling features ------------------------------------------------
_LAG_RE   = re.compile(r'(?i)lag[_\-]?(\d+)')
_RMEAN_RE = re.compile(r'(?i)rmean[_\-]?(\d+)')
_RSTD_RE  = re.compile(r'(?i)rstd[_\-]?(\d+)')

# -----------------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------------

def _hours_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return int(round((b - a).total_seconds() / 3600.0))

def ensure_nonempty_features(X: pd.DataFrame) -> pd.DataFrame:
    """If X has no columns, add a constant so estimators can train/predict."""
    if X.shape[1] == 0:
        X = X.copy()
        X["__const__"] = 1.0
    return X

def _fillna_for_selection(X: pd.DataFrame) -> pd.DataFrame:
    """Basic imputation for selector routines that cannot handle NaNs (selector-only)."""
    if X.shape[0] == 0 or X.shape[1] == 0:
        return X
    # median over numeric columns; non-numeric are skipped by numeric_only=True
    return X.fillna(X.median(numeric_only=True))

def evaluate_predictions(
    y_true, y_pred, *, seasonality: Optional[int] = None, eps: float = 1e-8
) -> Dict[str, float]:
    """Returns a set of regression metrics; RMSE is the one optimized in the tuner."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]
    if y_true.size == 0:
        raise ValueError("No valid pairs after filtering NaNs/Infs.")

    err = y_pred - y_true
    abs_err = np.abs(err)
    mae = float(abs_err.mean())
    rmse = float(np.sqrt(np.mean(err ** 2)))
    r2 = float(1.0 - (np.sum(err ** 2) / (np.sum((y_true - y_true.mean()) ** 2) + eps)))

    mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100.0)
    smape = float(np.nanmean(2.0 * abs_err / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100.0)
    wmape = float((np.sum(abs_err) / (np.sum(np.abs(y_true)) + eps)) * 100.0)

    mase = math.nan
    if seasonality and y_true.size > seasonality:
        naive_abs_err = np.abs(y_true[seasonality:] - y_true[:-seasonality])
        mase = float(mae / (naive_abs_err.mean() + eps))

    return {
        "n": float(y_true.size),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE%": mape,
        "SMAPE%": smape,
        "WMAPE%": wmape,
        "MASE": mase,
    }

# -----------------------------------------------------------------------------------
# Feature metadata & selection
# -----------------------------------------------------------------------------------

def infer_lag_meta(feature_cols: Sequence[str],
                   targets: Sequence[str]) -> Dict[str, Dict]:
    """
    Map feature -> {'owner':'<target or unknown>', 'kind':'lag|rmean|rstd', 'lag_h' or 'win_h'}.
    Owner is inferred by substring match to any target name (case-insensitive).
    """
    t_lower = [(t, t.lower()) for t in targets]
    meta: Dict[str, Dict] = {}
    for c in feature_cols:
        cl = c.lower()
        owner = "unknown"
        for t, tl in t_lower:
            base = tl.split("_")[0]
            if tl in cl or cl.startswith(base) or f"_{base}_" in cl:
                owner = t
                break
        m = _LAG_RE.search(cl)
        if m:
            meta[c] = {"owner": owner, "kind": "lag", "lag_h": int(m.group(1))}
            continue
        m = _RMEAN_RE.search(cl)
        if m:
            meta[c] = {"owner": owner, "kind": "rmean", "win_h": int(m.group(1))}
            continue
        m = _RSTD_RE.search(cl)
        if m:
            meta[c] = {"owner": owner, "kind": "rstd", "win_h": int(m.group(1))}
            continue
    return meta

def _build_sfs_estimator(name: str, use_gpu: bool = True) -> Any:
    """Lightweight base estimators for SFS (GPU-aware with safe fallbacks)."""
    name = (name or "ridge").lower()

    if name == "ridge":
        return Ridge(alpha=1.0)

    if name == "hgb":
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=RANDOM_STATE)

    if name == "gbr":
        return GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=RANDOM_STATE)

    if name == "lgbm" and LGBMRegressor is not None:
        est = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                            random_state=RANDOM_STATE, verbose=-1)
        if use_gpu:
            try:
                est.set_params(device="gpu")
            except Exception:
                try:
                    est.set_params(device_type="gpu")
                except Exception:
                    pass  # fallback to CPU silently
        return est

    if name == "xgb" and XGBRegressor is not None:
        est = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                           subsample=0.9, colsample_bytree=0.9,
                           objective="reg:squarederror", n_jobs=-1,
                           random_state=RANDOM_STATE)
        if use_gpu:
            # Prefer modern 'device' API, then fallback to 'gpu_hist'
            try:
                est.set_params(device="cuda")
            except Exception:
                try:
                    est.set_params(tree_method="gpu_hist", predictor="gpu_predictor")
                except Exception:
                    pass
        else:
            try:
                est.set_params(tree_method="hist")
            except Exception:
                pass
        return est

    if name == "cat" and CatBoostRegressor is not None:
        est = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6,
                                random_seed=RANDOM_STATE, loss_function="RMSE", verbose=False)
        if use_gpu:
            try:
                est.set_params(task_type="GPU")
            except Exception:
                pass
        return est

    # default
    return Ridge(alpha=1.0)

def select_columns(selector: str,
                   Xtr: pd.DataFrame,
                   ytr: np.ndarray,
                   *,
                   k: Optional[int] = None,
                   sfs_direction: str = "forward",
                   sfs_cv: int = 3,
                   sfs_base_estimator: str = "ridge",
                   sfs_use_gpu: bool = True) -> List[str]:
    """
    Leakage-safe selection run *only on the training window* Xtr/ytr.

    selector:
      - "all": keep all columns
      - "mi_top_k": top-k by mutual information (k default ~25% of features)
      - "sfs": sklearn SequentialFeatureSelector with TimeSeriesSplit CV. Use `sfs_direction` to pick "forward"/"backward".
      - Backward-compatible aliases "sfs_forward"/"sfs_backward" are accepted.

    Returns list of selected column names (subset of Xtr.columns).
    """
    if Xtr is None or len(Xtr.columns) == 0:
        return []

    if selector in (None, "all"):
        return list(Xtr.columns)

    Xs = _fillna_for_selection(Xtr)
    ys = np.asarray(ytr, dtype=float)

    if selector == "mi_top_k":
       # Robust MI: OHE non-numeric, then aggregate MI back to source feature
       Xnum = Xs.select_dtypes(include=[np.number])
       Xcat = Xs.select_dtypes(exclude=[np.number])
       if not Xcat.empty:
           Xcat_ohe = pd.get_dummies(Xcat, dummy_na=False)
           # Map dummy -> original feature
           back_map = {}
           for c in Xcat_ohe.columns:
               # conservative back-map: split on first '_' produced by get_dummies
               src = c.split('_', 1)[0]
               back_map[c] = src
           Xmi = pd.concat([Xnum, Xcat_ohe], axis=1)
       else:
           Xmi = Xnum
           back_map = {c: c for c in Xmi.columns}

       mi = mutual_info_regression(Xmi, ys, random_state=RANDOM_STATE)
       # Aggregate MI per original feature (max over its dummy columns)
       agg = {}
       for c, s in zip(Xmi.columns, mi):
           src = back_map.get(c, c)
           agg[src] = max(agg.get(src, 0.0), float(s))

       k_final = int(k) if k is not None else max(4, int(len(Xs.columns) * 0.25))
       k_final = max(1, min(k_final, len(Xs.columns)))
       # preserve original column order for tie-breaking
       ordered = sorted(Xs.columns, key=lambda c: agg.get(c, 0.0))
       return list(ordered[-k_final:])

    if selector in ("sfs", "sfs_forward", "sfs_backward"):
        direction = sfs_direction.lower()
        if selector.endswith("_forward"):
            direction = "forward"
        elif selector.endswith("_backward"):
            direction = "backward"
        if direction not in ("forward", "backward"):
            direction = "forward"

        k_final = int(k) if k is not None else max(5, int(len(Xs.columns) * 0.20))
        k_final = max(1, min(k_final, len(Xs.columns)))
        n_splits = max(2, int(sfs_cv))
        try:
            base_est = _build_sfs_estimator(sfs_base_estimator, use_gpu=sfs_use_gpu)
            tscv = TimeSeriesSplit(n_splits=n_splits)

            # Custom RMSE scorer that guarantees named DataFrame goes into predict()
            def _sfs_rmse_scorer(est, X, y):
                # If the estimator was fitted with names (has feature_names_in_ or LGBM booster),
                # give it a DataFrame; otherwise pass a plain ndarray to avoid the warning.
                if hasattr(est, "feature_names_in_") or hasattr(est, "booster_"):
                    X_in = ensure_named_df(X, like_est=est)
                else:
                    X_in = np.asarray(X)
                y_pred = np.asarray(est.predict(X_in), dtype=float).ravel()
                y_true = np.asarray(y, dtype=float).ravel()
                err = y_pred - y_true
                rmse = float((np.mean(err ** 2)) ** 0.5)
                return -rmse  # scikit-learn maximizes the score

            sfs = SequentialFeatureSelector(
                base_est,
                n_features_to_select=k_final,
                direction=direction,
                scoring=_sfs_rmse_scorer,  # key change
                cv=tscv,
                n_jobs=1
            )
            # ---- SFS safety: encode categoricals to numeric codes (1:1 columns, no OHE) ----
            Xs_sfs = Xs.copy()
            # Promote object/bool + low-card ints to category so codes are global (not per fold)
            for c in Xs_sfs.columns:
                s = Xs_sfs[c]
                if pd.api.types.is_object_dtype(s) or pd.api.types.is_bool_dtype(s):
                    Xs_sfs[c] = s.astype("category")
                elif pd.api.types.is_integer_dtype(s):
                    nunique = s.nunique(dropna=True)
                    if nunique <= max(24, int(0.02 * len(s))):
                        Xs_sfs[c] = s.astype("category")
            # Replace categories with codes; NaN becomes -1 → set back to NaN for numeric models
            for c in Xs_sfs.columns:
                if isinstance(Xs_sfs[c].dtype, pd.CategoricalDtype):
                    codes = Xs_sfs[c].cat.codes.astype("float32")
                    codes = codes.replace(-1, np.nan)  # keep NaN semantics
                    Xs_sfs[c] = codes

            # For estimators that dislike NaN (e.g., Ridge/GBR if used), impute numerics
            if sfs_base_estimator.lower() in {"ridge", "gbr"}:
                Xs_sfs = Xs_sfs.fillna(Xs_sfs.median(numeric_only=True))

            cols_all = list(Xs_sfs.columns)
            # Let SFS do its usual slicing; our scorer wraps them back with names.
            sfs.fit(Xs_sfs, ys)
            mask = sfs.get_support()
            cols = [cols_all[i] for i, keep in enumerate(mask) if keep]
            if not cols:
                mi = mutual_info_regression(Xs, ys, random_state=RANDOM_STATE)
                order = np.argsort(mi)[-k_final:]
                return list(Xs.columns[order])
            return cols
        except Exception as ex:
            warnings.warn(f"SFS failed with {ex}; falling back to MI top-k.")
            Xnum = Xs.select_dtypes(include=[np.number])
            Xcat = Xs.select_dtypes(exclude=[np.number])
            if not Xcat.empty:
                Xcat_ohe = pd.get_dummies(Xcat, dummy_na=False)
                back_map = {c: c.split('_', 1)[0] for c in Xcat_ohe.columns}
                Xmi = pd.concat([Xnum, Xcat_ohe], axis=1)
            else:
                Xmi = Xnum
                back_map = {c: c for c in Xmi.columns}
            mi = mutual_info_regression(Xmi, ys, random_state=RANDOM_STATE)
            agg = {}
            for c, s in zip(Xmi.columns, mi):
                src = back_map.get(c, c)
                agg[src] = max(agg.get(src, 0.0), float(s))
            ordered = sorted(Xs.columns, key=lambda c: agg.get(c, 0.0))
            return list(ordered[-k_final:])

    raise ValueError(f"Unknown selector: {selector}")

# -----------------------------------------------------------------------------------
# Split handling
# -----------------------------------------------------------------------------------

def cv_pairs_from_splits(index: pd.DatetimeIndex, splits_subset: List[Dict]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Turn rolling split dicts into (train_idx, valid_idx) pairs."""
    pairs = []
    for s in splits_subset:
        train_mask = (index >= pd.Timestamp(s["train_start"])) & (index <= pd.Timestamp(s["train_end"]))
        valid_mask = (index >= pd.Timestamp(s["test_start"]))  & (index <= pd.Timestamp(s["test_end"]))
        pairs.append((np.flatnonzero(train_mask), np.flatnonzero(valid_mask)))
    return pairs

# -----------------------------------------------------------------------------------
# Lag policy
# -----------------------------------------------------------------------------------

def choose_features_for_split(
    feature_cols: List[str],
    lag_meta: Dict[str, Dict],
    *,
    target: str,
    other_targets: Sequence[str],
    split: Dict,
    lag_policy: str,               # 'drop' | 'own' | 'mutual'
    keep_rolling_stats: bool = True,
    min_rolling_win: int = 1,
    allow_any_lags: bool = True
) -> Tuple[List[str], List[str], int]:
    """
    Returns (allowed_cols, recalc_cols, min_required_lag_hours).

    - 'drop': allow non-lag; allow only lags >= (gap + horizon); rollings are DROPPED regardless of keep_rolling_stats.
    - 'own':  allow all own-target lags and (optionally) rollings; mark short ones for rebuild.
              Cross-target: allow only safe long lags; cross rollings dropped.
    - 'mutual': treat own and cross short lags as rebuildable; cross rollings dropped.
    - Rolling filtering (pre-policy, except 'drop' which drops all rollings anyway):
        * if keep_rolling_stats=False: all rmean/rstd dropped;
        * else keep only rmean/rstd with win_h >= min_rolling_win.
    - If allow_any_lags=False: drop ALL lag_* features regardless of policy/length.
    """
    assert lag_policy in ("drop", "own", "mutual")
    gap_h     = _hours_between(pd.Timestamp(split["train_end"]), pd.Timestamp(split["test_start"]))
    horizon_h = _hours_between(pd.Timestamp(split["test_start"]), pd.Timestamp(split["test_end"]))
    min_req_h = gap_h + horizon_h

    allowed, recalc = [], []
    others = set(other_targets)

    for c in feature_cols:
        meta = lag_meta.get(c)
        if meta is None:
            allowed.append(c)  # non-lag/non-rolling
            continue

        owner = meta["owner"]
        kind  = meta["kind"]

        # Drop all lags if requested
        if kind == "lag" and not allow_any_lags:
            continue

        if lag_policy == "drop":
            # Under 'drop', never rebuild; keep only safe long lags; always drop rollings
            if kind in ("rmean", "rstd"):
                continue
            if kind == "lag" and meta["lag_h"] >= min_req_h:
                allowed.append(c)
            elif kind not in ("lag", "rmean", "rstd"):
                allowed.append(c)
            continue

        # Rolling controls for 'own'/'mutual'
        if kind in ("rmean", "rstd"):
            if not keep_rolling_stats:
                continue
            if int(meta.get("win_h", 0)) < int(min_rolling_win):
                continue

        treat_as_rebuildable = (owner == target) or (lag_policy == "mutual" and owner in others)
        if treat_as_rebuildable:
            allowed.append(c)
            if (kind == "lag" and meta["lag_h"] < min_req_h) or (kind in ("rmean", "rstd")):
                recalc.append(c)
        else:
            if kind == "lag" and meta["lag_h"] >= min_req_h:
                allowed.append(c)
            # cross rolling -> drop
    return allowed, recalc, min_req_h

# -----------------------------------------------------------------------------------
# Observed-or-predicted value helpers
# -----------------------------------------------------------------------------------

def _y_or_pred(owner: str,
               ts: pd.Timestamp,
               y_map: Dict[str, pd.Series],
               train_end: pd.Timestamp,
               preds_map: Dict[str, Dict[pd.Timestamp, float]]) -> float:
    if ts > train_end:
        return preds_map.get(owner, {}).get(ts, np.nan)
    return float(y_map[owner].get(ts, np.nan))

def _rolling(owner: str,
             t: pd.Timestamp,
             win_h: int,
             y_map: Dict[str, pd.Series],
             train_end: pd.Timestamp,
             preds_map: Dict[str, Dict[pd.Timestamp, float]],
             stat: str) -> float:
    vals = []
    for k in range(1, win_h + 1):
        src_ts = t - pd.Timedelta(hours=k)
        vals.append(_y_or_pred(owner, src_ts, y_map, train_end, preds_map))
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    if stat == "mean":
        return float(a.mean())
    return float(a.std(ddof=1)) if a.size >= 2 else np.nan

# -----------------------------------------------------------------------------------
# Categorical utilities + model-safe preprocessing
# -----------------------------------------------------------------------------------

def classify_features(
    df: pd.DataFrame,
    targets: Sequence[str],
    *,
    known_categorical: Optional[Sequence[str]] = None,
    max_cardinality: int = 24,
) -> tuple[list[str], list[str]]:
    """Return (num_cols, cat_cols) using simple, robust heuristics."""
    feature_cols = [c for c in df.columns if c not in targets]
    known = set(map(str, known_categorical or []))
    cat_cols, num_cols = [], []
    for c in feature_cols:
        s = df[c]
        if (c in known or isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(s) or pd.api.types.is_bool_dtype(s)):
            cat_cols.append(c); continue
        if pd.api.types.is_integer_dtype(s):
            nunique = s.nunique(dropna=True)
            if nunique <= max(max_cardinality, int(0.02 * len(s))):
                cat_cols.append(c); continue
        num_cols.append(c)
    return num_cols, cat_cols


def prepare_X_for_model(
    model_name: str,
    X: pd.DataFrame,
    categorical_cols: list[str] | None = None,
    xgb_native_categorical: bool = False,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """
    Normalize feature dtypes for each model family.

    Returns:
      X_out: dataframe ready for estimator.fit(...)
      cat_cols_out: list of categorical column names kept as category (for LGBM/XGB)
      feature_names: ordered list of column names in X_out
      ohe_cols_created: list of dummy columns created (for OHE branches)
    """
    model = (model_name or "").upper()
    X_out = X.copy()
    ohe_cols_created: list[str] = []

    if categorical_cols is None:
        categorical_cols = []

    # Promote known/object/bool to pandas 'category'
    cat_cols_out: list[str] = []
    for c in X_out.columns:
        s = X_out[c]
        is_cat = (
            c in categorical_cols
            or isinstance(s.dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(s)
            or pd.api.types.is_bool_dtype(s)
        )
        if is_cat:
            if not isinstance(s.dtype, pd.CategoricalDtype):
                X_out[c] = s.astype("category")
            cat_cols_out.append(c)

    if model == "LGBM":
        for c in X_out.select_dtypes(include=["float64"]).columns:
            X_out[c] = X_out[c].astype(np.float32)
        for c in X_out.select_dtypes(include=["int64"]).columns:
            X_out[c] = X_out[c].astype(np.int32)
        return X_out, cat_cols_out, list(X_out.columns), ohe_cols_created

    if model == "XGB":
        for c in X_out.select_dtypes(include=["float64"]).columns:
            X_out[c] = X_out[c].astype(np.float32)
        for c in X_out.select_dtypes(include=["int64"]).columns:
            X_out[c] = X_out[c].astype(np.int32)
        if xgb_native_categorical:
            return X_out, cat_cols_out, list(X_out.columns), ohe_cols_created
        else:
            before = set(X_out.columns)
            if cat_cols_out:
                X_out = pd.get_dummies(X_out, columns=cat_cols_out, dummy_na=False)
            ohe_cols_created = [c for c in X_out.columns if c not in before]
            return X_out.astype(np.float32), [], list(X_out.columns), ohe_cols_created

    if model == "CATBOOST":
        cat_set = set(categorical_cols or [])
        for c in X_out.columns:
            if c in cat_set:
                X_out[c] = X_out[c].astype("string").fillna("__nan__").astype(object)
        for c in X_out.select_dtypes(include=["float64"]).columns:
            X_out[c] = X_out[c].astype(np.float32)
        for c in X_out.select_dtypes(include=["int64"]).columns:
            X_out[c] = X_out[c].astype(np.int32)
        return X_out, cat_cols_out, list(X_out.columns), ohe_cols_created

    if model in {"HGBR", "GBR"}:
        before = set(X_out.columns)
        if cat_cols_out:
            X_out = pd.get_dummies(X_out, columns=cat_cols_out, dummy_na=False)
        ohe_cols_created = [c for c in X_out.columns if c not in before]
        for c in X_out.select_dtypes(include=["float64"]).columns:
            X_out[c] = X_out[c].astype(np.float32)
        for c in X_out.select_dtypes(include=["int64"]).columns:
            X_out[c] = X_out[c].astype(np.int32)
        return X_out, [], list(X_out.columns), ohe_cols_created

    return X_out, cat_cols_out, list(X_out.columns), ohe_cols_created

def freeze_categories(X: pd.DataFrame, cat_levels: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Ensure category codes match training by fixing category levels.
    """
    if not cat_levels:
        return X
    X = X.copy()
    for c, levels in cat_levels.items():
        if c in X.columns and isinstance(X[c].dtype, pd.CategoricalDtype):
            X[c] = X[c].cat.set_categories(levels)
    return X

def align_to_columns(X: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Ensure columns match training design matrix (use NaN to preserve dtypes)."""
    return X.reindex(columns=columns)  # fill with NaN (safer for categorical))

def safe_for_model(X: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    GBR cannot handle NaN; others (LGBM/XGB/CatBoost/HGBR) can.
    For GBR we:
      - use keep_empty_features=True so columns aren't dropped when all-NaN
      - then replace any remaining NaNs (e.g., from all-NaN cols) with 0.0
    """
    if model_name.upper() == "GBR":
        # Keep shape even if a column is entirely NaN
        try:
            imp = SimpleImputer(strategy="median", keep_empty_features=True)
        except TypeError:
            # Older sklearn without keep_empty_features: do a two-step fallback
            imp = SimpleImputer(strategy="median")
            X_imp = imp.fit_transform(X.fillna(np.nan))
            # If shape changed due to dropped all-NaN columns, rebuild with manual fill
            if X_imp.shape[1] != X.shape[1]:
                X_filled = X.copy()
                # fill all-NaN cols with 0, others stay NaN for median impute
                all_nan_cols = X.columns[X.isna().all(axis=0)]
                if len(all_nan_cols):
                    X_filled[all_nan_cols] = 0.0
                imp2 = SimpleImputer(strategy="median")
                X_imp = imp2.fit_transform(X_filled)
                X_imp = np.asarray(X_imp, dtype=float)
                return pd.DataFrame(X_imp, index=X.index, columns=X.columns)
            X_imp = np.asarray(X_imp, dtype=float)
            # replace any residual NaNs
            X_imp[np.isnan(X_imp)] = 0.0
            return pd.DataFrame(X_imp, index=X.index, columns=X.columns)

        # Modern path with keep_empty_features
        X_imp = imp.fit_transform(X)
        X_imp = np.asarray(X_imp, dtype=float)
        # Replace residual NaNs caused by all-NaN columns (median undefined)
        X_imp[np.isnan(X_imp)] = 0.0
        return pd.DataFrame(X_imp, index=X.index, columns=X.columns)

    return X

def ensure_named_df(
    X,
    *,
    columns_hint: list[str] | None = None,
    like_est: Any | None = None
) -> pd.DataFrame:
    """
    Return X as a pandas.DataFrame with explicit column names.
    """
    if isinstance(X, pd.DataFrame):
        return X
    X_arr = np.asarray(X)

    cols = None
    if columns_hint is not None:
        cols = list(columns_hint)
    elif like_est is not None:
        if hasattr(like_est, "feature_names_in_"):
            cols = list(getattr(like_est, "feature_names_in_"))
        elif hasattr(like_est, "booster_"):
            try:
                cols = list(like_est.booster_.feature_name())
            except Exception:
                cols = None

    if cols is None:
        cols = [f"col_{i}" for i in range(X_arr.shape[1])]

    return pd.DataFrame(X_arr, columns=cols)
# -----------------------------------------------------------------------------------
# Universal rolling (any number of targets)
# -----------------------------------------------------------------------------------

def roll_predict_multi(
    est_map: Dict[str, object],
    df_features: pd.DataFrame,
    *,
    targets: Sequence[str],
    allowed_map: Dict[str, List[str]],
    recalc_map: Dict[str, List[str]],
    lag_meta: Dict[str, Dict],
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    y_map: Dict[str, pd.Series],
    model_name: str,
    cat_cols_map: Dict[str, List[str]],
    trained_cols_map: Dict[str, List[str]],
    cat_levels_map: Dict[str, Dict[str, List[str]]],
    ohe_cols_map: Dict[str, List[str]],
    xgb_native_categorical: bool = True
) -> Dict[str, Dict[pd.Timestamp, float]]:
    """Roll 1-step through (train_end..test_end] and predict all targets together with per-model transforms."""
    mask = (df_features.index > train_end) & (df_features.index <= test_end)
    roll_idx = df_features.index[mask]
    preds = {t: {} for t in targets}
    model = (model_name or "").upper()

    for t in roll_idx:
        rows: Dict[str, pd.DataFrame] = {}
        for tgt in targets:
            cols = allowed_map.get(tgt, [])
            row = df_features.loc[t, cols].copy() if cols else pd.Series(dtype=float)
            for c in recalc_map.get(tgt, []):
                meta = lag_meta[c]; owner = meta["owner"]; kind = meta["kind"]
                if kind == "lag":
                    src = t - pd.Timedelta(hours=meta["lag_h"])
                    row[c] = _y_or_pred(owner, src, y_map, train_end, preds)
                elif kind == "rmean":
                    row[c] = _rolling(owner, t, meta["win_h"], y_map, train_end, preds, "mean")
                elif kind == "rstd":
                    row[c] = _rolling(owner, t, meta["win_h"], y_map, train_end, preds, "std")

            df_row = ensure_nonempty_features(pd.DataFrame([row]))
            df_row, _, _, _ = prepare_X_for_model(model, df_row, cat_cols_map.get(tgt, []),
                                                  xgb_native_categorical=xgb_native_categorical)
            # Freeze categorical levels to training
            df_row = freeze_categories(df_row, cat_levels_map.get(tgt, {}))
            # Align and model-specific NaN handling
            df_row = align_to_columns(df_row, trained_cols_map[tgt])

            # zero-fill only OHE dummies if this model uses OHE
            if model in {"HGBR", "GBR"} or (model == "XGB" and not xgb_native_categorical):
                ohe_cols = [c for c in ohe_cols_map.get(tgt, []) if c in df_row.columns]
                if ohe_cols:
                    df_row[ohe_cols] = df_row[ohe_cols].fillna(0.0)

            df_row = safe_for_model(df_row, model)
            rows[tgt] = df_row

        for tgt in targets:
            Xrow = ensure_named_df(rows[tgt], columns_hint=trained_cols_map[tgt], like_est=est_map[tgt])
            preds[tgt][t] = float(est_map[tgt].predict(Xrow)[0])
    return preds

def build_validation_matrices(
    df_features: pd.DataFrame,
    *,
    targets: Sequence[str],
    val_index: pd.DatetimeIndex,
    allowed_map: Dict[str, List[str]],
    recalc_map: Dict[str, List[str]],
    lag_meta: Dict[str, Dict],
    train_end: pd.Timestamp,
    y_map: Dict[str, pd.Series],
    preds_history: Dict[str, Dict[pd.Timestamp, float]],
    model_name: str,
    cat_cols_map: Dict[str, List[str]],
    trained_cols_map: Dict[str, List[str]],
    cat_levels_map: Dict[str, Dict[str, List[str]]],
    ohe_cols_map: Dict[str, List[str]],
    xgb_native_categorical: bool = True
) -> Dict[str, pd.DataFrame]:
    """Rebuild X_val using observed/predicted values as needed + per-model transforms."""
    Xval = {}
    model = (model_name or "").upper()
    for tgt in targets:
        cols = allowed_map.get(tgt, [])
        Xv = df_features.loc[val_index, cols].copy() if cols else pd.DataFrame(index=val_index)
        for t in val_index:
            for c in recalc_map.get(tgt, []):
                meta = lag_meta[c]; owner = meta["owner"]; kind = meta["kind"]
                if kind == "lag":
                    src = t - pd.Timedelta(hours=meta["lag_h"])
                    Xv.at[t, c] = _y_or_pred(owner, src, y_map, train_end, preds_history)
                elif kind == "rmean":
                    Xv.at[t, c] = _rolling(owner, t, meta["win_h"], y_map, train_end, preds_history, "mean")
                elif kind == "rstd":
                    Xv.at[t, c] = _rolling(owner, t, meta["win_h"], y_map, train_end, preds_history, "std")
        Xv = ensure_nonempty_features(Xv)
        Xv, _, _, _ = prepare_X_for_model(model, Xv, cat_cols_map.get(tgt, []),
                                          xgb_native_categorical=xgb_native_categorical)
        Xv = freeze_categories(Xv, cat_levels_map.get(tgt, {}))
        Xv = align_to_columns(Xv, trained_cols_map[tgt])

        if model in {"HGBR", "GBR"} or (model == "XGB" and not xgb_native_categorical):
            ohe_cols = [c for c in ohe_cols_map.get(tgt, []) if c in Xv.columns]
            if ohe_cols:
                Xv[ohe_cols] = Xv[ohe_cols].fillna(0.0)

        Xv = safe_for_model(Xv, model)
        Xval[tgt] = Xv
    return Xval

# -----------------------------------------------------------------------------------
# Models & Optuna spaces
# -----------------------------------------------------------------------------------

def default_params(model_name: str) -> Dict[str, Any]:
    """Reasonable defaults when warm-starting peer models without a study."""
    if model_name == "LGBM":
        return dict(
            n_estimators=800, learning_rate=0.05, num_leaves=63, max_depth=-1,
            min_child_samples=40, subsample=0.9, colsample_bytree=0.9,
            reg_alpha=0.0, reg_lambda=0.0, random_state=RANDOM_STATE
        )
    if model_name == "XGB":
        return dict(
            n_estimators=1200, learning_rate=0.05, max_depth=6, min_child_weight=1.0,
            subsample=0.9, colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=1.0,
            random_state=RANDOM_STATE, objective="reg:squarederror", n_jobs=-1
        )
    if model_name == "CatBoost":
        return dict(
            iterations=1500, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
            bagging_temperature=0.0, random_seed=RANDOM_STATE, loss_function="RMSE", verbose=False
        )
    if model_name == "HGBR":
        return dict(learning_rate=0.05, max_iter=800, max_leaf_nodes=63, l2_regularization=0.0, random_state=RANDOM_STATE)
    if model_name == "GBR":
        return dict(n_estimators=800, learning_rate=0.05, max_depth=3, subsample=0.9, random_state=RANDOM_STATE)
    raise ValueError(f"Unknown model: {model_name}")

def suggest_params(trial: optuna.trial.Trial, model_name: str, *, prefix: str = "") -> Dict[str, Any]:
    """
    Returns an *unprefixed* dict of model params. Internally uses prefixed keys in Optuna
    so we can propose different params per target in the same trial if needed.
    """
    def _i(name: str, low: int, high: int, log: bool = False) -> int:
        key = f"{prefix}{name}"
        return trial.suggest_int(key, low, high, log=log)

    def _f(name: str, low: float, high: float, log: bool = False) -> float:
        key = f"{prefix}{name}"
        return trial.suggest_float(key, low, high, log=log)

    if model_name == "LGBM":
        return dict(
            n_estimators       = _i("n_estimators", 400, 3000, log=True),
            learning_rate      = _f("learning_rate", 1e-3, 0.2, log=True),
            num_leaves         = _i("num_leaves", 15, 255, log=True),
            max_depth          = _i("max_depth", -1, 16),
            min_child_samples  = _i("min_child_samples", 5, 300),
            subsample          = _f("subsample", 0.5, 1.0),
            colsample_bytree   = _f("colsample_bytree", 0.5, 1.0),
            reg_alpha          = _f("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda         = _f("reg_lambda", 1e-8, 10.0, log=True),
            random_state       = RANDOM_STATE,
        )
    if model_name == "XGB":
        return dict(
            n_estimators       = _i("n_estimators", 400, 4000, log=True),
            learning_rate      = _f("learning_rate", 1e-3, 0.3, log=True),
            max_depth          = _i("max_depth", 3, 16),
            min_child_weight   = _f("min_child_weight", 0.1, 20.0, log=True),
            subsample          = _f("subsample", 0.5, 1.0),
            colsample_bytree   = _f("colsample_bytree", 0.5, 1.0),
            reg_alpha          = _f("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda         = _f("reg_lambda", 1e-8, 10.0, log=True),
            random_state       = RANDOM_STATE,
            objective          = "reg:squarederror",
            n_jobs             = -1,
        )
    if model_name == "CatBoost":
        return dict(
            iterations          = _i("iterations", 500, 3000, log=True),
            learning_rate       = _f("learning_rate", 1e-3, 0.2, log=True),
            depth               = _i("depth", 4, 10),
            l2_leaf_reg         = _f("l2_leaf_reg", 1e-2, 50.0, log=True),
            bagging_temperature = _f("bagging_temperature", 0.0, 5.0),
            random_seed         = RANDOM_STATE,
            loss_function       = "RMSE",
            verbose             = False,
        )
    if model_name == "HGBR":
        return dict(
            learning_rate       = _f("learning_rate", 1e-3, 0.3, log=True),
            max_iter            = _i("max_iter", 200, 2000, log=True),
            max_leaf_nodes      = _i("max_leaf_nodes", 15, 255),
            l2_regularization   = _f("l2_regularization", 1e-9, 10.0, log=True),
            random_state        = RANDOM_STATE,
        )
    if model_name == "GBR":
        return dict(
            n_estimators        = _i("n_estimators", 200, 2000, log=True),
            learning_rate       = _f("learning_rate", 1e-3, 0.3, log=True),
            max_depth           = _i("max_depth", 2, 6),
            subsample           = _f("subsample", 0.5, 1.0),
            random_state        = RANDOM_STATE,
        )
    raise ValueError(f"Unknown model: {model_name}")

def build_estimator(model_name: str, params: Dict, use_gpu: bool = True):
    """Instantiate estimator with safe device flags if available in this environment."""
    name = (model_name or "").upper()
    if name == "LGBM" and LGBMRegressor is not None:
        est = LGBMRegressor(**params, verbose=-1)
        try:
            if use_gpu:
                for k in ("device", "device_type"):
                    try:
                        est.set_params(**{k: "gpu"})
                        break
                    except Exception:
                        pass
            else:
                for k in ("device", "device_type"):
                    try:
                        est.set_params(**{k: "cpu"})
                        break
                    except Exception:
                        pass
        except Exception:
            pass
        return est
    if name == "XGB" and XGBRegressor is not None:
        est = XGBRegressor(**params)
        try:
            est.set_params(device=("cuda" if use_gpu else "cpu"))
        except Exception:
            try:
                est.set_params(tree_method=("gpu_hist" if use_gpu else "hist"))
                if use_gpu:
                    est.set_params(predictor="gpu_predictor")
            except Exception:
                pass
        return est
    if name == "CATBOOST":
        if CatBoostRegressor is None:
            raise RuntimeError("CatBoost not installed.")
        est = CatBoostRegressor(**params)
        try:
            est.set_params(task_type=("GPU" if use_gpu else "CPU"))
        except Exception:
            pass
        return est
    if name == "HGBR":
        return HistGradientBoostingRegressor(**params)
    if name == "GBR":
        return GradientBoostingRegressor(**params)
    raise ValueError(f"Unknown or unavailable model: {model_name}")

# -----------------------------------------------------------------------------------
# CV reporting helper (fixed to use fold-fitted models) + fresh clone
# -----------------------------------------------------------------------------------

def fresh_clone(est):
    """Create a fresh, unfitted estimator instance with the same hyperparameters."""
    cls = est.__class__
    if hasattr(est, "get_params"):
        return cls(**est.get_params(deep=True))
    return copy.deepcopy(est)

def _cv_scores_per_target_fixed(
    est_map: Dict[str, Any],
    df: pd.DataFrame,
    X_all: pd.DataFrame,
    y_all: Dict[str, np.ndarray],
    y_map: Dict[str, pd.Series],
    targets: Sequence[str],
    pairs: List[Tuple[np.ndarray, np.ndarray]],
    splits_use: List[Dict],
    allowed_map: Dict[str, List[str]],
    recalc_map: Dict[str, List[str]],
    lag_meta: Dict[str, Dict],
    seasonality: int,
    lag_policy: str,
    model_name: str,
    known_categorical: Optional[Sequence[str]],
    xgb_native_categorical: bool
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {t: {"RMSE": 0.0, "MAPE%": 0.0} for t in targets}
    counts = 0
    model = (model_name or "").upper()

    # categorical discovery once
    _, cat_cols_all = classify_features(df, targets, known_categorical=known_categorical)

    for (tr_idx, va_idx), split in zip(pairs, splits_use):
        tmp = {t: fresh_clone(est_map[t]) for t in targets}
        cat_cols_map: Dict[str, List[str]] = {}
        trained_cols_map: Dict[str, List[str]] = {}
        cat_idx_map: Dict[str, Optional[List[int]]] = {}
        cat_levels_map: Dict[str, Dict[str, List[str]]] = {}
        ohe_cols_map: Dict[str, List[str]] = {}

        # fit fold models
        for t in targets:
            cols = allowed_map[t]
            base_tr = ensure_nonempty_features(X_all.iloc[tr_idx][cols])
            cat_cols_t = [c for c in cols if c in cat_cols_all]
            Xtr, cat_names, trained_cols, ohe_cols = prepare_X_for_model(
                model, base_tr, cat_cols_t, xgb_native_categorical=xgb_native_categorical
            )
            Xtr = safe_for_model(Xtr, model)

            cat_cols_map[t] = cat_cols_t
            trained_cols_map[t] = trained_cols
            ohe_cols_map[t] = ohe_cols
            # indices for CatBoost; names for freeze map
            cat_idx_t = [Xtr.columns.get_loc(c) for c in cat_names]
            cat_idx_map[t] = cat_idx_t
            cat_levels_map[t] = {c: list(Xtr[c].cat.categories)
                                 for c in cat_names if isinstance(Xtr[c].dtype, pd.CategoricalDtype)}

            tmp_est = tmp[t]
            if model == "CATBOOST":
                tmp_est.fit(Xtr, y_all[t][tr_idx], cat_features=cat_idx_t, verbose=False)
            else:
                tmp_est.fit(Xtr, y_all[t][tr_idx])

        # build validation
        if lag_policy == "drop":
            Xval = {}
            for t in targets:
                cols = allowed_map[t]
                base_va = ensure_nonempty_features(X_all.iloc[va_idx][cols])
                base_va, _, _, _ = prepare_X_for_model(model, base_va, cat_cols_map[t], xgb_native_categorical)
                base_va = freeze_categories(base_va, cat_levels_map.get(t, {}))
                base_va = align_to_columns(base_va, trained_cols_map[t])

                if model in {"HGBR", "GBR"} or (model == "XGB" and not xgb_native_categorical):
                    ohe_cols = [c for c in ohe_cols_map.get(t, []) if c in base_va.columns]
                    if ohe_cols:
                        base_va[ohe_cols] = base_va[ohe_cols].fillna(0.0)

                base_va = safe_for_model(base_va, model)
                Xval[t] = base_va
        else:
            preds_hist = roll_predict_multi(
                tmp, df_features=X_all, targets=targets,
                allowed_map=allowed_map, recalc_map=recalc_map, lag_meta=lag_meta,
                train_end=pd.Timestamp(split["train_end"]),
                test_start=pd.Timestamp(split["test_start"]), test_end=pd.Timestamp(split["test_end"]),
                y_map=y_map,
                model_name=model, cat_cols_map=cat_cols_map, trained_cols_map=trained_cols_map,
                cat_levels_map=cat_levels_map, ohe_cols_map=ohe_cols_map,
                xgb_native_categorical=xgb_native_categorical
            )
            Xval = build_validation_matrices(
                X_all, targets=targets, val_index=X_all.index[va_idx],
                allowed_map=allowed_map, recalc_map=recalc_map, lag_meta=lag_meta,
                train_end=pd.Timestamp(split["train_end"]), y_map=y_map, preds_history=preds_hist,
                model_name=model, cat_cols_map=cat_cols_map, trained_cols_map=trained_cols_map,
                cat_levels_map=cat_levels_map, ohe_cols_map=ohe_cols_map,
                xgb_native_categorical=xgb_native_categorical
            )

        for t in targets:
            Xv = ensure_named_df(Xval[t], columns_hint=trained_cols_map[t], like_est=tmp[t])
            yhat = tmp[t].predict(Xv)
            m = evaluate_predictions(y_all[t][va_idx], yhat, seasonality=seasonality)
            out[t]["RMSE"]  += float(m["RMSE"])
            out[t]["MAPE%"] += float(m["MAPE%"])
        counts += 1

    if counts > 0:
        for t in targets:
            out[t]["RMSE"]  /= counts
            out[t]["MAPE%"] /= counts
    return out

# -----------------------------------------------------------------------------------
# Shared objective factory (single study for both targets)
# -----------------------------------------------------------------------------------

def _make_objective_shared(
    df: pd.DataFrame,
    pairs: List[Tuple[np.ndarray, np.ndarray]],
    splits_use: List[Dict],
    targets: Sequence[str],
    feature_cols: List[str],
    lag_meta: Dict[str, Dict],
    model_name: str,
    use_gpu: bool,
    lag_policy: str,
    feature_selector: str,
    k_features: Optional[int],
    sfs_direction: str,
    sfs_cv: int,
    sfs_base_estimator: str,
    keep_rolling_stats: bool,
    min_rolling_win: int,
    seasonality: int,
    target_weights: Optional[Dict[str, float]] = None,
    agg: str = "mean",  # "mean" | "weighted" | "max"
    allow_any_lags: bool = True,
    known_categorical: Optional[Sequence[str]] = None,
    xgb_native_categorical: bool = True
):
    X_all = df[feature_cols]
    y_map = {t: df[t] for t in targets}
    y_all = {t: df[t].values for t in targets}
    weights = target_weights or {t: 1.0 for t in targets}
    _, cat_cols_all = classify_features(df, targets, known_categorical=known_categorical)

    # Stable early stopping behavior – pass ES to fit(...), never mutate a fitted model.
    def _fit_with_optional_es(est, Xtr, ytr, Xva=None, yva=None, cat_idx=None):
        has_valid = Xva is not None and yva is not None
        if model_name == "LGBM" and has_valid and lgb is not None:
            try:
                est.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="rmse",
                        callbacks=[lgb.early_stopping(50, first_metric_only=True)])
            except Exception:
                est.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="rmse", early_stopping_rounds=50)
        elif model_name == "XGB":
            fit_kwargs = {}
            if has_valid:
                fit_kwargs["eval_set"] = [(Xva, yva)]
            try:
                est.set_params(eval_metric="rmse", verbosity=0)
            except Exception:
                pass
            # enable native categoricals + pick GPU/CPU tree method correctly
            try:
                est.set_params(enable_categorical=xgb_native_categorical)
                if xgb_native_categorical:
                    if use_gpu:
                        # prefer modern device API, then fallback to gpu_hist flags
                        try:
                            est.set_params(device="cuda")
                        except Exception:
                            pass
                        try:
                            est.set_params(tree_method="gpu_hist", predictor="gpu_predictor")
                        except Exception:
                            est.set_params(tree_method="gpu_hist")
                    else:
                        est.set_params(tree_method="hist")
            except Exception:
                pass
            est.fit(Xtr, ytr, verbose=False, **fit_kwargs)
        elif model_name == "CatBoost":
            fit_kwargs = dict(verbose=False)
            if has_valid:
                fit_kwargs.update(eval_set=(Xva, yva), use_best_model=True, early_stopping_rounds=50)
            est.fit(Xtr, ytr, cat_features=cat_idx, **fit_kwargs)
        else:
            est.fit(Xtr, ytr)

    def _aggregate(rmses: Dict[str, float]) -> float:
        if agg == "mean":
            return float(np.mean(list(rmses.values())))
        if agg == "weighted":
            s = 0.0; w = 0.0
            for t, v in rmses.items():
                s += weights.get(t, 1.0) * v
                w += weights.get(t, 1.0)
            return float(s / max(w, 1e-9))
        if agg == "max":
            return float(max(rmses.values()))
        return float(np.mean(list(rmses.values())))

    def objective(trial: optuna.trial.Trial) -> float:
        params = suggest_params(trial, model_name)
        est_template = {t: build_estimator(model_name, params, use_gpu=use_gpu) for t in targets}

        fold_scores = []
        for (tr_idx, va_idx), split in zip(pairs, splits_use):
            est_map_fold = {t: fresh_clone(est_template[t]) for t in targets}

            allowed_map: Dict[str, List[str]] = {}
            recalc_map: Dict[str, List[str]] = {}
            cat_cols_map: Dict[str, List[str]] = {}
            trained_cols_map: Dict[str, List[str]] = {}
            cat_idx_map: Dict[str, Optional[List[int]]] = {}
            cat_levels_map: Dict[str, Dict[str, List[str]]] = {}
            ohe_cols_map: Dict[str, List[str]] = {}

            for t in targets:
                allowed, recalc, _ = choose_features_for_split(
                    feature_cols, lag_meta, target=t,
                    other_targets=[x for x in targets if x != t],
                    split=split, lag_policy=lag_policy,
                    keep_rolling_stats=keep_rolling_stats, min_rolling_win=min_rolling_win,
                    allow_any_lags=allow_any_lags
                )
                Xtr_allowed = X_all.iloc[tr_idx][allowed]
                cols = select_columns(
                    feature_selector, Xtr_allowed, y_all[t][tr_idx],
                    k=k_features, sfs_direction=sfs_direction, sfs_cv=sfs_cv,
                    sfs_base_estimator=sfs_base_estimator, sfs_use_gpu=use_gpu
                )
                allowed_map[t] = cols
                recalc_map[t]  = [c for c in recalc if c in cols]
                cat_cols_t = [c for c in cols if c in cat_cols_all]
                cat_cols_map[t] = cat_cols_t

            # fit
            for t in targets:
                Xtr_base = ensure_nonempty_features(X_all.iloc[tr_idx][allowed_map[t]])
                model = (model_name or "").upper()
                Xtr, cat_names, trained_cols, ohe_cols = prepare_X_for_model(
                    model, Xtr_base, cat_cols_map[t], xgb_native_categorical=xgb_native_categorical
                )
                Xtr = safe_for_model(Xtr, model)
                trained_cols_map[t] = trained_cols
                ohe_cols_map[t] = ohe_cols
                # indices for CatBoost
                cat_idx_t = [Xtr.columns.get_loc(c) for c in cat_names]
                cat_idx_map[t] = cat_idx_t
                cat_levels_map[t] = {c: list(Xtr[c].cat.categories)
                     for c in cat_names if isinstance(Xtr[c].dtype, pd.CategoricalDtype)}

                if lag_policy == "drop":
                    Xva_base = ensure_nonempty_features(X_all.iloc[va_idx][allowed_map[t]])
                    Xva, _, _, _ = prepare_X_for_model(model_name, Xva_base, cat_cols_map[t], xgb_native_categorical=xgb_native_categorical)
                    Xva = freeze_categories(Xva, cat_levels_map.get(t, {}))
                    Xva = align_to_columns(Xva, trained_cols)
                    Xva = safe_for_model(Xva, model_name)
                    _fit_with_optional_es(est_map_fold[t], Xtr, y_all[t][tr_idx], Xva, y_all[t][va_idx], cat_idx=cat_idx_t)
                else:
                    _fit_with_optional_es(est_map_fold[t], Xtr, y_all[t][tr_idx], cat_idx=cat_idx_t)

            # build validation matrices
            if lag_policy == "drop":
                Xval = {}
                for t in targets:
                    base_va = ensure_nonempty_features(X_all.iloc[va_idx][allowed_map[t]])
                    base_va, _, _, _ = prepare_X_for_model(model_name, base_va, cat_cols_map[t], xgb_native_categorical)
                    base_va = freeze_categories(base_va, cat_levels_map.get(t, {}))
                    base_va = align_to_columns(base_va, trained_cols_map[t])
                    if model_name in {"HGBR", "GBR"} or (model_name == "XGB" and not xgb_native_categorical):
                        ohe_cols = [c for c in ohe_cols_map.get(t, []) if c in base_va.columns]
                        if ohe_cols:
                            base_va[ohe_cols] = base_va[ohe_cols].fillna(0.0)
                    base_va = safe_for_model(base_va, model_name)
                    Xval[t] = base_va
            else:
                preds_hist = roll_predict_multi(
                    est_map_fold, df_features=X_all, targets=targets,
                    allowed_map=allowed_map, recalc_map=recalc_map, lag_meta=lag_meta,
                    train_end=pd.Timestamp(split["train_end"]),
                    test_start=pd.Timestamp(split["test_start"]),
                    test_end=pd.Timestamp(split["test_end"]),
                    y_map=y_map,
                    model_name=model_name, cat_cols_map=cat_cols_map, trained_cols_map=trained_cols_map,
                    cat_levels_map=cat_levels_map, ohe_cols_map=ohe_cols_map,
                    xgb_native_categorical=xgb_native_categorical
                )
                Xval = build_validation_matrices(
                    X_all, targets=targets, val_index=X_all.index[va_idx],
                    allowed_map=allowed_map, recalc_map=recalc_map, lag_meta=lag_meta,
                    train_end=pd.Timestamp(split["train_end"]), y_map=y_map, preds_history=preds_hist,
                    model_name=model_name, cat_cols_map=cat_cols_map, trained_cols_map=trained_cols_map,
                    cat_levels_map=cat_levels_map, ohe_cols_map=ohe_cols_map,
                    xgb_native_categorical=xgb_native_categorical
                )

            rmses: Dict[str, float] = {}
            for t in targets:
                Xv = ensure_named_df(Xval[t], columns_hint=trained_cols_map[t], like_est=est_map_fold[t])
                yhat = est_map_fold[t].predict(Xv)
                m = evaluate_predictions(y_true=y_all[t][va_idx], y_pred=yhat, seasonality=seasonality)
                rmses[t] = float(m["RMSE"])
            fold_scores.append(_aggregate(rmses))

            trial.report(float(np.mean(fold_scores)), step=len(fold_scores)-1)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(fold_scores))

    return objective

# -----------------------------------------------------------------------------------
# Separate two-study optimization with peer predictions (alternating)
# -----------------------------------------------------------------------------------

def _make_objective_single_target_given_peer(
    df: pd.DataFrame,
    pairs: List[Tuple[np.ndarray, np.ndarray]],
    splits_use: List[Dict],
    targets: Sequence[str],
    t_main: str,
    peer_params: Dict[str, Any],      # fixed hyperparams for the peer target
    feature_cols: List[str],
    lag_meta: Dict[str, Dict],
    model_name: str,
    use_gpu: bool,
    lag_policy: str,
    feature_selector: str,
    k_features: Optional[int],
    sfs_direction: str,
    sfs_cv: int,
    sfs_base_estimator: str,
    keep_rolling_stats: bool,
    min_rolling_win: int,
    seasonality: int,
    allow_any_lags: bool = True,
    known_categorical: Optional[Sequence[str]] = None,
    xgb_native_categorical: bool = True
):
    """Optimize t_main while using a fixed peer model for mutual lag recursion."""
    assert len(targets) == 2, "This helper currently supports exactly two targets."
    t_peer = [t for t in targets if t != t_main][0]

    X_all = df[feature_cols]
    y_map = {t: df[t] for t in targets}
    y_all = {t: df[t].values for t in targets}
    _, cat_cols_all = classify_features(df, targets, known_categorical=known_categorical)

    def _fit_with_optional_es(est, Xtr, ytr, Xva=None, yva=None, cat_idx=None):
        has_valid = Xva is not None and yva is not None
        if model_name == "LGBM" and has_valid and lgb is not None:
            try:
                est.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="rmse",
                        callbacks=[lgb.early_stopping(50, first_metric_only=True)])
            except Exception:
                est.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="rmse", early_stopping_rounds=50)
        elif model_name == "XGB":
            fit_kwargs = {}
            if has_valid:
                fit_kwargs["eval_set"] = [(Xva, yva)]
            try:
                est.set_params(eval_metric="rmse", verbosity=0)
            except Exception:
                pass
            # enable native categoricals + pick GPU/CPU tree method correctly
            try:
                est.set_params(enable_categorical=xgb_native_categorical)
                if xgb_native_categorical:
                    if use_gpu:
                        # prefer modern device API, then fallback to gpu_hist flags
                        try:
                            est.set_params(device="cuda")
                        except Exception:
                            pass
                        try:
                            est.set_params(tree_method="gpu_hist", predictor="gpu_predictor")
                        except Exception:
                            est.set_params(tree_method="gpu_hist")
                    else:
                        est.set_params(tree_method="hist")
            except Exception:
                pass
            est.fit(Xtr, ytr, verbose=False, **fit_kwargs)
        elif model_name == "CatBoost":
            fit_kwargs = dict(verbose=False)
            if has_valid:
                fit_kwargs.update(eval_set=(Xva, yva), use_best_model=True, early_stopping_rounds=50)
            est.fit(Xtr, ytr, cat_features=cat_idx, **fit_kwargs)
        else:
            est.fit(Xtr, ytr)

    def objective(trial: optuna.trial.Trial) -> float:
        params_main = suggest_params(trial, model_name)
        est_main_template = build_estimator(model_name, params_main, use_gpu=use_gpu)
        est_peer_template = build_estimator(model_name, peer_params, use_gpu=use_gpu)

        fold_scores = []
        for (tr_idx, va_idx), split in zip(pairs, splits_use):
            est_main = fresh_clone(est_main_template)
            est_peer = fresh_clone(est_peer_template)

            allowed_map: Dict[str, List[str]] = {}
            recalc_map: Dict[str, List[str]] = {}
            cat_cols_map: Dict[str, List[str]] = {}
            trained_cols_map: Dict[str, List[str]] = {}
            cat_idx_map: Dict[str, Optional[List[int]]] = {}
            cat_levels_map: Dict[str, Dict[str, List[str]]] = {}
            ohe_cols_map: Dict[str, List[str]] = {}

            for t in targets:
                allowed, recalc, _ = choose_features_for_split(
                    feature_cols, lag_meta, target=t,
                    other_targets=[x for x in targets if x != t],
                    split=split, lag_policy=lag_policy,
                    keep_rolling_stats=keep_rolling_stats, min_rolling_win=min_rolling_win,
                    allow_any_lags=allow_any_lags
                )
                Xtr_allowed = X_all.iloc[tr_idx][allowed]
                cols = select_columns(
                    feature_selector, Xtr_allowed, y_all[t][tr_idx],
                    k=k_features, sfs_direction=sfs_direction, sfs_cv=sfs_cv,
                    sfs_base_estimator=sfs_base_estimator, sfs_use_gpu=use_gpu
                )
                allowed_map[t] = cols
                recalc_map[t]  = [c for c in recalc if c in cols]
                cat_cols_map[t] = [c for c in cols if c in cat_cols_all]

            # Fit main
            Xtr_main_base = ensure_nonempty_features(X_all.iloc[tr_idx][allowed_map[t_main]])
            Xtr_main, cat_names_main, trained_cols_main, ohe_cols_main = prepare_X_for_model(
                model_name, Xtr_main_base, cat_cols_map[t_main], xgb_native_categorical=xgb_native_categorical
            )
            Xtr_main = safe_for_model(Xtr_main, model_name)
            trained_cols_map[t_main] = trained_cols_main
            cat_idx_main = [Xtr_main.columns.get_loc(c) for c in cat_names_main]
            cat_idx_map[t_main] = cat_idx_main

            if lag_policy == "drop":
                Xva_main_base = ensure_nonempty_features(X_all.iloc[va_idx][allowed_map[t_main]])
                Xva_main, _, _, _ = prepare_X_for_model(model_name, Xva_main_base, cat_cols_map[t_main], xgb_native_categorical)
                Xva_main = align_to_columns(Xva_main, trained_cols_main)
                Xva_main = safe_for_model(Xva_main, model_name)
                _fit_with_optional_es(est_main, Xtr_main, y_all[t_main][tr_idx], Xva_main, y_all[t_main][va_idx], cat_idx=cat_idx_main)
            else:
                _fit_with_optional_es(est_main, Xtr_main, y_all[t_main][tr_idx], cat_idx=cat_idx_main)

            # Fit peer
            Xtr_peer_base = ensure_nonempty_features(X_all.iloc[tr_idx][allowed_map[t_peer]])
            Xtr_peer, cat_names_peer, trained_cols_peer, ohe_cols_peer = prepare_X_for_model(
                model_name, Xtr_peer_base, cat_cols_map[t_peer], xgb_native_categorical=xgb_native_categorical
            )
            Xtr_peer = safe_for_model(Xtr_peer, model_name)
            trained_cols_map[t_peer] = trained_cols_peer
            cat_idx_peer = [Xtr_peer.columns.get_loc(c) for c in cat_names_peer]
            cat_idx_map[t_peer] = cat_idx_peer

            if lag_policy == "drop":
                Xva_peer_base = ensure_nonempty_features(X_all.iloc[va_idx][allowed_map[t_peer]])
                Xva_peer, _, _, _ = prepare_X_for_model(model_name, Xva_peer_base, cat_cols_map[t_peer], xgb_native_categorical)
                Xva_peer = align_to_columns(Xva_peer, trained_cols_peer)
                Xva_peer = safe_for_model(Xva_peer, model_name)
                _fit_with_optional_es(est_peer, Xtr_peer, y_all[t_peer][tr_idx], Xva_peer, y_all[t_peer][va_idx], cat_idx=cat_idx_peer)
            else:
                _fit_with_optional_es(est_peer, Xtr_peer, y_all[t_peer][tr_idx], cat_idx=cat_idx_peer)

            # Validation matrices
            if lag_policy == "drop":
                base_va = ensure_nonempty_features(X_all.iloc[va_idx][allowed_map[t_main]])
                base_va, _, _, _ = prepare_X_for_model(model_name, base_va, cat_cols_map[t_main], xgb_native_categorical)
                base_va = freeze_categories(base_va, cat_levels_map.get(t_main, {}))
                base_va = align_to_columns(base_va, trained_cols_main)
                if model_name in {"HGBR", "GBR"} or (model_name == "XGB" and not xgb_native_categorical):
                    ohe_cols = [c for c in ohe_cols_main if c in base_va.columns]
                    if ohe_cols:
                        base_va[ohe_cols] = base_va[ohe_cols].fillna(0.0)
                base_va = safe_for_model(base_va, model_name)
                Xval = {t_main: base_va}
            else:
                est_map_fold = {t_main: est_main, t_peer: est_peer}
                preds_hist = roll_predict_multi(
                    est_map_fold, df_features=X_all, targets=targets,
                    allowed_map=allowed_map, recalc_map=recalc_map, lag_meta=lag_meta,
                    train_end=pd.Timestamp(split["train_end"]),
                    test_start=pd.Timestamp(split["test_start"]),
                    test_end=pd.Timestamp(split["test_end"]),
                    y_map=y_map,
                    model_name=model_name, cat_cols_map=cat_cols_map, trained_cols_map=trained_cols_map,
                    cat_levels_map=cat_levels_map, ohe_cols_map=ohe_cols_map,
                    xgb_native_categorical=xgb_native_categorical
                )
                Xval = build_validation_matrices(
                    X_all, targets=targets, val_index=X_all.index[va_idx],
                    allowed_map=allowed_map, recalc_map=recalc_map, lag_meta=lag_meta,
                    train_end=pd.Timestamp(split["train_end"]), y_map=y_map, preds_history=preds_hist,
                    model_name=model_name, cat_cols_map=cat_cols_map, trained_cols_map=trained_cols_map,
                    cat_levels_map=cat_levels_map, ohe_cols_map=ohe_cols_map,
                    xgb_native_categorical=xgb_native_categorical
                )

            Xv = ensure_named_df(Xval[t_main], columns_hint=trained_cols_map[t_main], like_est=est_main)
            yhat = est_main.predict(Xv)
            m = evaluate_predictions(y_true=y_all[t_main][va_idx], y_pred=yhat, seasonality=seasonality)
            fold_scores.append(float(m["RMSE"]))

            trial.report(float(np.mean(fold_scores)), step=len(fold_scores)-1)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(fold_scores))

    return objective

# -----------------------------------------------------------------------------------
# Top-level tuner orchestrator
# -----------------------------------------------------------------------------------

def tune_multi_targets(
    df: pd.DataFrame,
    splits: List[Dict],
    *,
    targets: Sequence[str] = ("P_Power", "Q_Power"),
    model_name: str = "LGBM",
    n_trials: int = 50,
    use_gpu: bool = True,
    max_splits: Optional[int] = None,
    seasonality: int = 24,
    storage: Optional[str] = None,
    lag_policy: str = "drop",          # 'drop' | 'own' | 'mutual'
    share_hyperparams: bool = True,    # True = one study; False = two studies with optional alternation
    separate_strategy: str = "auto",   # 'auto' | 'independent' | 'alternating'
    outer_loops: int = 2,              # for 'alternating': how many passes P<-Q<-P...
    peer_warmstart_trials: int = 0,    # small warmstart study for peer before alternating (0 to skip)
    feature_selector: str = "mi_top_k",# "all" | "mi_top_k" | "sfs"
    k_features: Optional[int] = None,
    sfs_direction: str = "forward",
    sfs_cv: int = 3,
    sfs_base_estimator: str = "ridge",
    keep_rolling_stats: bool = True,
    min_rolling_win: int = 1,
    allow_any_lags: bool = True,
    coupled_agg: str = "mean",
    target_weights: Optional[Dict[str, float]] = None,
    study_name_prefix: Optional[str] = None,
    known_categorical: Optional[Sequence[str]] = None,
    xgb_native_categorical: bool = True
):
    """
    Single entry point with categorical-awareness:
      - If share_hyperparams=True -> one Optuna study across both targets; objective aggregates their RMSEs.
      - If share_hyperparams=False -> two Optuna studies:
            - 'independent'  : each target optimized alone.
            - 'alternating'  : coordinate-descent style; while optimizing one target, the peer model with its current
                               best hyperparams is refit per fold to provide predictions for mutual lags.
      - Feature selection is always performed on the *train* window per fold (leakage-safe).
      - Final feature set per target is selected on the last split's *train* window only; models are refit on full data.
    """
    if isinstance(targets, (str, np.str_)):
        targets = [targets]
    else:
        targets = list(targets)
    
    # Optional: fail fast if a target is missing from df
    missing = [t for t in targets if t not in df.columns]
    if missing:
        raise ValueError(f"Targets not found in df: {missing}")
    feature_cols = [c for c in df.columns if c not in targets]
    X_all = df[feature_cols]
    y_map = {t: df[t] for t in targets}
    y_all = {t: df[t].values for t in targets}

    splits_use = splits if (max_splits is None) else splits[:max_splits]
    pairs = cv_pairs_from_splits(df.index, splits_use)
    lag_meta = infer_lag_meta(feature_cols, targets)

    # Storage/Study names
    def _mk_name(base: str):
        return f"{study_name_prefix+'_' if study_name_prefix else ''}{model_name}_{base}_{lag_policy}"

    # ------------------ Shared coupled study ------------------
    if share_hyperparams:
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            storage=storage,
            study_name=_mk_name(f"SHARED_{coupled_agg.upper()}"),
            load_if_exists=bool(storage)
        )
        objective = _make_objective_shared(
            df=df, pairs=pairs, splits_use=splits_use, targets=targets,
            feature_cols=feature_cols, lag_meta=lag_meta, model_name=model_name, use_gpu=use_gpu,
            lag_policy=lag_policy, feature_selector=feature_selector, k_features=k_features,
            sfs_direction=sfs_direction, sfs_cv=sfs_cv, sfs_base_estimator=sfs_base_estimator,
            keep_rolling_stats=keep_rolling_stats, min_rolling_win=min_rolling_win, seasonality=seasonality,
            target_weights=target_weights, agg=coupled_agg, allow_any_lags=allow_any_lags,
            known_categorical=known_categorical, xgb_native_categorical=xgb_native_categorical
        )
        study.optimize(objective, n_trials=n_trials, n_jobs=1)

        best_params_all = study.best_params
        params_shared = suggest_params(optuna.trial.FixedTrial(best_params_all), model_name)
        est_map = {t: build_estimator(model_name, params_shared, use_gpu=use_gpu) for t in targets}

    # ------------------ Separate studies ------------------
    else:
        if separate_strategy == "auto":
            separate_strategy = "alternating" if lag_policy == "mutual" else "independent"

        best_params_map: Dict[str, Dict[str, Any]] = {t: default_params(model_name) for t in targets}
        best_study_map: Dict[str, optuna.Study] = {}

        # Optional warm-start peer quick studies
        if peer_warmstart_trials and separate_strategy == "alternating":
            for t in targets:
                peer = [x for x in targets if x != t][0]
                study_peer = optuna.create_study(
                    direction="minimize",
                    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
                    pruner=optuna.pruners.MedianPruner(n_startup_trials=3),
                    storage=None,
                    study_name=_mk_name(f"WARM_{peer}")
                )
                objective_peer = _make_objective_shared(
                    df=df, pairs=pairs, splits_use=splits_use, targets=[peer,],  # single target
                    feature_cols=feature_cols, lag_meta=lag_meta, model_name=model_name, use_gpu=use_gpu,
                    lag_policy=lag_policy if lag_policy != "mutual" else "own",  # safe warmstart
                    feature_selector=feature_selector, k_features=k_features,
                    sfs_direction=sfs_direction, sfs_cv=sfs_cv, sfs_base_estimator=sfs_base_estimator,
                    keep_rolling_stats=keep_rolling_stats, min_rolling_win=min_rolling_win, seasonality=seasonality,
                    target_weights=None, agg="mean", allow_any_lags=allow_any_lags,
                    known_categorical=known_categorical, xgb_native_categorical=xgb_native_categorical
                )
                study_peer.optimize(objective_peer, n_trials=int(peer_warmstart_trials), n_jobs=1)
                best_params_map[peer] = suggest_params(optuna.trial.FixedTrial(study_peer.best_params), model_name)
        # Decide how many outer loops to run based on strategy
        if separate_strategy == "alternating":
            outer_loops_use = int(max(1, outer_loops))
        else:
            outer_loops_use = 1  # single pass for 'independent
        # Outer coordinate-descent style loops
        for outer in range(outer_loops_use):
            iter_tag = f"_iter{outer+1}" if separate_strategy == "alternating" else ""          
            for t_main in targets:
                t_peer = [x for x in targets if x != t_main]#[0]                                
                if t_peer:
                    t_peer = t_peer[0]                

                study_t = optuna.create_study(
                    direction="minimize",
                    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
                    storage=storage,
                    study_name=_mk_name(f"{t_main}_SEPARATE_{separate_strategy.upper()}{iter_tag}"),
                    load_if_exists=False
                )
                if separate_strategy == "independent":
                    objective_t = _make_objective_shared(
                        df=df, pairs=pairs, splits_use=splits_use, targets=[t_main,],
                        feature_cols=feature_cols, lag_meta=lag_meta, model_name=model_name, use_gpu=use_gpu,
                        lag_policy=lag_policy, feature_selector=feature_selector, k_features=k_features,
                        sfs_direction=sfs_direction, sfs_cv=sfs_cv, sfs_base_estimator=sfs_base_estimator,
                        keep_rolling_stats=keep_rolling_stats, min_rolling_win=min_rolling_win, seasonality=seasonality,
                        target_weights=None, agg="mean", allow_any_lags=allow_any_lags,
                        known_categorical=known_categorical, xgb_native_categorical=xgb_native_categorical
                    )
                else:
                    objective_t = _make_objective_single_target_given_peer(
                        df=df, pairs=pairs, splits_use=splits_use, targets=targets, t_main=t_main,
                        peer_params=best_params_map[t_peer], feature_cols=feature_cols, lag_meta=lag_meta,
                        model_name=model_name, use_gpu=use_gpu, lag_policy=lag_policy,
                        feature_selector=feature_selector, k_features=k_features,
                        sfs_direction=sfs_direction, sfs_cv=sfs_cv, sfs_base_estimator=sfs_base_estimator,
                        keep_rolling_stats=keep_rolling_stats, min_rolling_win=min_rolling_win, seasonality=seasonality,
                        allow_any_lags=allow_any_lags, known_categorical=known_categorical,
                        xgb_native_categorical=xgb_native_categorical
                    )
                study_t.optimize(objective_t, n_trials=n_trials, n_jobs=1)
                best_study_map[t_main] = study_t
                best_params_map[t_main] = suggest_params(optuna.trial.FixedTrial(study_t.best_params), model_name)

        # Build final estimators from best params per target
        est_map = {t: build_estimator(model_name, best_params_map[t], use_gpu=use_gpu) for t in targets}

        # Combine study handles for return (choose last ones)
        study = best_study_map.get(targets[0], None)

    # ---- Refit best on full data and assemble metadata ---------------------------------------
    last_split = splits_use[-1]
    allowed_map_final: Dict[str, List[str]] = {}
    recalc_map_final: Dict[str, List[str]] = {}

    last_train_mask = (df.index >= pd.Timestamp(last_split["train_start"])) & (df.index <= pd.Timestamp(last_split["train_end"]))
    tr_idx_final = np.flatnonzero(last_train_mask)

    # Find categoricals (once)
    _, cat_cols_all = classify_features(df, targets, known_categorical=known_categorical)

    for t in targets:
        allowed, recalc, _ = choose_features_for_split(
            feature_cols, lag_meta, target=t,
            other_targets=[x for x in targets if x != t],
            split=last_split, lag_policy=lag_policy,
            keep_rolling_stats=keep_rolling_stats, min_rolling_win=min_rolling_win,
            allow_any_lags=allow_any_lags
        )
        cols = select_columns(
            feature_selector,
            X_all.iloc[tr_idx_final][allowed],
            y_all[t][tr_idx_final],
            k=k_features, sfs_direction=sfs_direction, sfs_cv=sfs_cv,
            sfs_base_estimator=sfs_base_estimator, sfs_use_gpu=use_gpu
        )
        allowed_map_final[t] = cols
        recalc_map_final[t]  = [c for c in recalc if c in cols]

        # Final refit on full data using the same preprocessing
        cat_cols_t = [c for c in cols if c in cat_cols_all]
        base_full = ensure_nonempty_features(X_all[cols])
        Xfull, cat_names_final, trained_cols_final, ohe_cols_final = prepare_X_for_model(
            model_name, base_full, cat_cols_t, xgb_native_categorical=xgb_native_categorical
        )
        Xfull = safe_for_model(Xfull, model_name)

        if model_name == "CatBoost":
            cat_idx_final = [Xfull.columns.get_loc(c) for c in cat_names_final] if cat_names_final else None
            est_map[t].fit(Xfull, y_all[t], cat_features=cat_idx_final, verbose=False)
        elif model_name == "XGB":
            try:
                est_map[t].set_params(enable_categorical=xgb_native_categorical)
                try:
                    est_map[t].set_params(device=("cuda" if use_gpu else "cpu"))
                except Exception:
                    est_map[t].set_params(tree_method=("gpu_hist" if use_gpu else "hist"))
                    if use_gpu:
                        est_map[t].set_params(predictor="gpu_predictor")
            except Exception:
                pass
            est_map[t].fit(Xfull, y_all[t])
        else:
            est_map[t].fit(Xfull, y_all[t])

    # ----- Per-target CV scores for reporting (not averaged) ----------------------------------
    cv_per_target = _cv_scores_per_target_fixed(
        est_map, df=df, X_all=X_all, y_all=y_all, y_map=y_map, targets=targets,
        pairs=pairs, splits_use=splits_use, allowed_map=allowed_map_final, recalc_map=recalc_map_final,
        lag_meta=lag_meta, seasonality=seasonality, lag_policy=lag_policy,
        model_name=model_name, known_categorical=known_categorical, xgb_native_categorical=xgb_native_categorical
    )

    metas: Dict[str, Any] = {}
    for t in targets:
        metas[t] = {
            "target": t,
            "model": model_name,
            "best_params": est_map[t].get_params() if hasattr(est_map[t], "get_params") else None,
            "lag_policy": lag_policy,
            "features": allowed_map_final[t],
            "recalc_features": recalc_map_final[t],
            "feature_selector": feature_selector,
            "k_features": int(k_features) if k_features is not None else None,
            "sfs_settings": {
                "direction": sfs_direction,
                "cv": int(sfs_cv),
                "base_estimator": sfs_base_estimator
            } if str(feature_selector).startswith("sfs") or feature_selector == "sfs" else None,
            "rolling_settings": {
                "keep_rolling_stats": bool(keep_rolling_stats),
                "min_rolling_win": int(min_rolling_win)
            },
            "cv_rmse": cv_per_target[t]["RMSE"],
            "cv_mape%": cv_per_target[t]["MAPE%"],
            "share_hyperparams": bool(share_hyperparams),
            "separate_strategy": separate_strategy if not share_hyperparams else None
        }
    return study, est_map, allowed_map_final, metas

# -----------------------------------------------------------------------------------
# Save / load
# -----------------------------------------------------------------------------------

def save_native(model, path, meta: dict | None = None):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    ext = ".joblib"
    if isinstance(model, (HistGradientBoostingRegressor, GradientBoostingRegressor)):
        joblib.dump(model, str(p.with_suffix(".joblib")), compress=3); ext = ".joblib"
    elif XGBRegressor is not None and isinstance(model, XGBRegressor):
        model.save_model(str(p.with_suffix(".json"))); ext = ".json"
    elif LGBMRegressor is not None and isinstance(model, LGBMRegressor):
        try:
            s = model.booster_.model_to_string()
            p.with_suffix(".txt").write_text(s, encoding="utf-8"); ext = ".txt"
        except Exception:
            joblib.dump(model, str(p.with_suffix(".joblib")), compress=3); ext = ".joblib"
    elif CatBoostRegressor is not None and isinstance(model, CatBoostRegressor):
        model.save_model(str(p.with_suffix(".cbm"))); ext = ".cbm"
    else:
        joblib.dump(model, str(p.with_suffix(".joblib")), compress=3); ext = ".joblib"
    if meta:
        (p.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return p.with_suffix(ext)

def load_native(path):
    p = Path(path); suf = p.suffix.lower()
    if suf == ".joblib": return joblib.load(p)
    if suf == ".json" and XGBRegressor is not None:
        m = XGBRegressor(); m.load_model(str(p)); return m
    if suf == ".txt" and lgb is not None:
        return lgb.Booster(model_file=str(p))
    if suf == ".cbm" and CatBoostRegressor is not None:
        m = CatBoostRegressor(); m.load_model(str(p)); return m
    raise ValueError(f"Unsupported extension: {suf}")

# -----------------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------------

def _read_pq_xlsx(xlsx_path: str) -> List[pd.DataFrame]:
    xls = pd.ExcelFile(xlsx_path)
    dfs: List[pd.DataFrame] = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
        if "Datetime" not in df.columns:
            for cand in ["datetime", "date", "timestamp"]:
                if cand in df.columns:
                    df.rename(columns={cand: "Datetime"}, inplace=True); break
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df.set_index("Datetime", inplace=True)
        df.sort_index(inplace=True)
        dfs.append(df)
    return dfs

def _read_splits_xlsx(xlsx_path: str) -> List[List[Dict[str, str]]]:
    """
    Returns a list (per-sheet) of list-of-dicts with keys:
    ['train_start','train_end','test_start','test_end'].
    """
    xls = pd.ExcelFile(xlsx_path)
    all_splits: List[List[Dict[str, str]]] = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
        df.columns = [str(c).strip().lower() for c in df.columns]
        cols = {}
        for c in df.columns:
            if "train" in c and "start" in c:
                cols[c] = "train_start"
            elif "train" in c and ("end" in c or "stop" in c):
                cols[c] = "train_end"
            elif "test" in c and "start" in c:
                cols[c] = "test_start"
            elif "test" in c and ("end" in c or "stop" in c):
                cols[c] = "test_end"
        if cols:
            df = df.rename(columns=cols)
        required = ["train_start","train_end","test_start","test_end"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Splits sheet '{sheet}' missing columns: {missing}")
        for c in required:
            df[c] = pd.to_datetime(df[c])
        sdicts = df[required].to_dict("records")
        all_splits.append(sdicts)
    return all_splits

# -----------------------------------------------------------------------------------
# Main (example CLI-style usage)
# -----------------------------------------------------------------------------------

if __name__ == "__main__":
    
    # 1) Read inputs from project space
    pq_path = rf"data\PQ.xlsx"
    splits_path = rf"data\splits.xlsx"
    out_dir = Path(rf"Output"); out_dir.mkdir(parents=True, exist_ok=True)
    
    PQ_list = _read_pq_xlsx(pq_path)
    splits_list = _read_splits_xlsx(splits_path)

    if len(PQ_list) != len(splits_list):
        warnings.warn(f"PQ sheets ({len(PQ_list)}) != splits sheets ({len(splits_list)}). Proceeding with min count.")
    n_pairs = min(len(PQ_list), len(splits_list))
    #print(PQ_list[0])
    # 2) Configure models and tuning options
    models_to_tune = ["CatBoost"] # "LGBM", "CatBoost", "XGB", "HGBR", "GBR"

    FEATURE_SELECTOR = "all"       # "all" | "mi_top_k" | "sfs"
    K_FEATURES = None              # amount of features for "mi_top_k" or "sfs"; None = all
    SFS_DIRECTION = "forward"      # "forward" or "backward"
    SFS_CV = 3                     # CV folds for SFS
    SFS_BASE = "cat"             # "ridge"|"hgb"|"gbr"|"lgbm"|"xgb"|"cat"

    KEEP_ROLLING_STATS = True     # False => drop all rmean/rstd (also dropped under lag_policy='drop') True
    MIN_ROLLING_WIN = 2            # e.g., 2 will drop rmean1/rstd1 but keep >=2
    ALLOW_ANY_LAGS = True          # set False to ban *all* lag_* features everywhere

    SHARE_HYPERPARAMS = True            # Shared study per model across both targets
    SEPARATE_STRATEGY = "auto"           # 'auto'|'independent'|'alternating'
    OUTER_LOOPS = 2
    PEER_WARM_TRIALS = 0

    # Known categorical columns in PQ.xlsx (adjust if you add/remove engineered features)
    KNOWN_CATS = [
        "Rainy","hour","weekday","season_idx","day_in_season","season_len",
        "is_holiday","is_day_before_hol","is_day_after_hol","is_weekend",
        "is_new_year","is_jan2","is_old_new_year","is_orthxmas","is_dec25"
    ]
    
    # 3) Run per sheet and per model (example: first sheet only to keep it quick)
    for name in models_to_tune:        
        for i in range(n_pairs): #n_pairs
            i=2
            print(f"study {name} {i} started")
            df = PQ_list[i]
            sdicts = splits_list[i]
            
            study, models, columns, metas = tune_multi_targets(
                df=df,
                splits=sdicts,
                targets=("P_Power","Q_Power"), #("P_Power","Q_Power")["P_Power"]
                model_name=name,
                n_trials=10,
                use_gpu=True,
                lag_policy="mutual",                    # "own" or "mutual" enable recursion, or "drop"
                share_hyperparams=SHARE_HYPERPARAMS,
                separate_strategy=SEPARATE_STRATEGY,
                outer_loops=OUTER_LOOPS,
                peer_warmstart_trials=PEER_WARM_TRIALS,
                feature_selector=FEATURE_SELECTOR,
                k_features=K_FEATURES,
                sfs_direction=SFS_DIRECTION,
                sfs_cv=SFS_CV,
                sfs_base_estimator=SFS_BASE,
                keep_rolling_stats=KEEP_ROLLING_STATS,
                min_rolling_win=MIN_ROLLING_WIN,
                allow_any_lags=ALLOW_ANY_LAGS,
                coupled_agg="mean",
                target_weights=None,
                study_name_prefix=f"sheet{i}",
                known_categorical=KNOWN_CATS,
                xgb_native_categorical=True
            )

            for tgt in ("P_Power","Q_Power"): #("P_Power","Q_Power") ["P_Power"]
                path = save_native(models[tgt], out_dir / f"Shared_mutual_lags_{name}_sheet{i}_{tgt}_best", meta=metas[tgt])



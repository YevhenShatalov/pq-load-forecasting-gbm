# -*- coding: utf-8 -*-
"""Tune and export P-Q forecasting models with chronological validation.

The module supports LightGBM, XGBoost, CatBoost, histogram-based gradient
boosting, and gradient boosting. It provides exogenous-only, own-history, and
mutual-history feature policies, optional feature selection, and resumable
Optuna checkpoints.

Run ``python "2. train_models_fixed.py" --help`` for command-line options.
Notebook users may edit and call ``run_from_main_settings()`` explicitly.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
SCRIPT_VERSION = "2.2"
np.random.seed(RANDOM_STATE)

# ---- Regex for lag/rolling features ------------------------------------------------
_LAG_RE   = re.compile(r'(?i)lag[_\-]?(\d+)')
_RMEAN_RE = re.compile(r'(?i)rmean[_\-]?(\d+)')
_RSTD_RE  = re.compile(r'(?i)rstd[_\-]?(\d+)')


def _json_default(value: Any):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding=encoding, newline="\n")
    os.replace(tmp, path)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def _atomic_joblib_dump(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    joblib.dump(value, tmp, compress=3)
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(
        Path(path),
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default),
    )


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def _stable_hash(payload: Any, *, length: Optional[int] = None) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    return digest[:length] if length else digest


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_seed(label: str) -> int:
    offset = int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)
    return int((RANDOM_STATE + offset) % (2**31 - 1))


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    return cleaned or "run"

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
    t_lower = sorted(((t, t.lower()) for t in targets), key=lambda item: len(item[1]), reverse=True)
    meta: Dict[str, Dict] = {}
    for c in feature_cols:
        cl = c.lower()
        owner = "unknown"
        for t, tl in t_lower:
            base = tl.split("_")[0]
            exact_target = tl in cl
            compact_dynamic = re.search(
                rf"(^|_){re.escape(base)}_(?:lag|rmean|rstd)[_\-]?\d+($|_)", cl
            )
            if exact_target or compact_dynamic:
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


def _selection_matrix(X: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, str]]:
    """Create a finite numeric matrix and an exact encoded-column source map."""
    numeric = X.select_dtypes(include=[np.number]).copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median(numeric_only=True)).fillna(0.0)
    parts = [numeric]
    back_map = {str(c): str(c) for c in numeric.columns}

    categorical = X.select_dtypes(exclude=[np.number])
    for source_index, source in enumerate(categorical.columns):
        dummy = pd.get_dummies(categorical[source], dummy_na=False, dtype=np.float32)
        names = [f"__cat_{source_index}_{j}" for j in range(dummy.shape[1])]
        dummy.columns = names
        parts.append(dummy)
        back_map.update({name: str(source) for name in names})

    matrix = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=X.index)
    if matrix.shape[1] == 0:
        matrix["__const__"] = 1.0
        back_map["__const__"] = "__const__"
    return matrix, back_map

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
                    pass  # Retain CPU settings.
        return est

    if name == "xgb" and XGBRegressor is not None:
        est = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                           subsample=0.9, colsample_bytree=0.9,
                           objective="reg:squarederror", n_jobs=-1,
                           random_state=RANDOM_STATE)
        if use_gpu:
            # Retain compatibility with both current and earlier XGBoost device APIs.
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

    # Fallback selector estimator.
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
       # Encode non-numeric columns, then aggregate scores to source features.
       Xmi, back_map = _selection_matrix(Xs)

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

            # Preserve feature names for estimators that validate input columns.
            def _sfs_rmse_scorer(est, X, y):
                # Use a DataFrame only when the fitted estimator retains feature names.
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
                scoring=_sfs_rmse_scorer,
                cv=tscv,
                n_jobs=1
            )
            # Encode categorical predictors without changing the feature count.
            Xs_sfs = Xs.copy()
            # Establish categorical levels before SFS creates its chronological folds.
            for c in Xs_sfs.columns:
                s = Xs_sfs[c]
                if pd.api.types.is_object_dtype(s) or pd.api.types.is_bool_dtype(s):
                    Xs_sfs[c] = s.astype("category")
                elif pd.api.types.is_integer_dtype(s):
                    nunique = s.nunique(dropna=True)
                    if nunique <= max(24, int(0.02 * len(s))):
                        Xs_sfs[c] = s.astype("category")
            # Convert categories to numeric codes while preserving missing values.
            for c in Xs_sfs.columns:
                if isinstance(Xs_sfs[c].dtype, pd.CategoricalDtype):
                    codes = Xs_sfs[c].cat.codes.astype("float32")
                    codes = codes.replace(-1, np.nan)
                    Xs_sfs[c] = codes

            # Ridge and GBR require finite numeric inputs.
            if sfs_base_estimator.lower() in {"ridge", "gbr"}:
                Xs_sfs = Xs_sfs.fillna(Xs_sfs.median(numeric_only=True)).fillna(0.0)

            cols_all = list(Xs_sfs.columns)
            # The scorer restores feature names after SFS column slicing.
            sfs.fit(Xs_sfs, ys)
            mask = sfs.get_support()
            cols = [cols_all[i] for i, keep in enumerate(mask) if keep]
            if not cols:
                Xmi, back_map = _selection_matrix(Xs)
                mi = mutual_info_regression(Xmi, ys, random_state=RANDOM_STATE)
                agg: Dict[str, float] = {}
                for c, score in zip(Xmi.columns, mi):
                    src = back_map.get(str(c), str(c))
                    agg[src] = max(agg.get(src, 0.0), float(score))
                ordered = sorted(Xs.columns, key=lambda c: agg.get(str(c), 0.0))
                return list(ordered[-k_final:])
            return cols
        except Exception as ex:
            warnings.warn(f"SFS failed with {ex}; falling back to MI top-k.")
            Xmi, back_map = _selection_matrix(Xs)
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

def select_splits_for_tuning(
    splits: Sequence[Dict],
    max_splits: Optional[int],
    *,
    strategy: str = "stratified",
) -> List[Dict]:
    """Select representative tuning folds without silently taking only early dates.

    ``stratified`` assigns approximately equal quotas to each split-generation
    scheme and samples each scheme over its full time range.  Use ``all`` (or
    ``max_splits=None``) to preserve every supplied fold.
    """
    records = [dict(item, _source_index=i) for i, item in enumerate(splits)]
    if not records:
        raise ValueError("No cross-validation splits were supplied.")
    if max_splits is None or int(max_splits) >= len(records) or strategy == "all":
        return records

    limit = int(max_splits)
    if limit < 1:
        raise ValueError("max_splits must be positive or None.")
    strategy = str(strategy).lower()
    if strategy == "first":
        return records[:limit]
    if strategy == "even":
        positions = np.linspace(0, len(records) - 1, num=limit)
        chosen = sorted({int(round(pos)) for pos in positions})
        return [records[i] for i in chosen]
    if strategy != "stratified":
        raise ValueError("split_selection must be one of: all, first, even, stratified")

    grouped: Dict[str, List[Dict]] = {}
    for item in records:
        grouped.setdefault(str(item.get("scheme") or "unspecified"), []).append(item)
    group_names = sorted(grouped)
    base_quota, remainder = divmod(limit, len(group_names))
    selected: List[Dict] = []
    for group_no, name in enumerate(group_names):
        group = sorted(grouped[name], key=lambda item: pd.Timestamp(item["test_start"]))
        quota = min(len(group), base_quota + (1 if group_no < remainder else 0))
        if quota:
            positions = np.linspace(0, len(group) - 1, num=quota)
            selected.extend(group[int(round(pos))] for pos in positions)

    selected_by_index = {int(item["_source_index"]): item for item in selected}
    if len(selected_by_index) < limit:
        remaining = [item for item in records if int(item["_source_index"]) not in selected_by_index]
        need = min(limit - len(selected_by_index), len(remaining))
        if need:
            positions = np.linspace(0, len(remaining) - 1, num=need)
            for pos in positions:
                item = remaining[int(round(pos))]
                selected_by_index[int(item["_source_index"])] = item
    return sorted(selected_by_index.values(), key=lambda item: pd.Timestamp(item["test_start"]))[:limit]


def _expected_hourly_index(start: Any, end: Any, *, label: str) -> pd.DatetimeIndex:
    """Return an inclusive hourly range and reject non-hourly boundaries."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if pd.isna(start_ts) or pd.isna(end_ts) or start_ts > end_ts:
        raise ValueError(f"Invalid {label} interval: {start_ts} .. {end_ts}")
    elapsed = end_ts - start_ts
    if elapsed % pd.Timedelta(hours=1) != pd.Timedelta(0):
        raise ValueError(f"{label} boundaries must lie on the same hourly grid.")
    return pd.date_range(start=start_ts, end=end_ts, freq="h")


def _recursive_index_from_history(
    index: pd.DatetimeIndex,
    *,
    history_end: Any,
    test_start: Any,
    test_end: Any,
) -> pd.DatetimeIndex:
    """Validate and return every recursive hour from the observation cutoff onward."""
    history_end_ts = pd.Timestamp(history_end)
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)
    if history_end_ts >= test_start_ts:
        raise ValueError(
            f"history_end must precede test_start: {history_end_ts} >= {test_start_ts}"
        )
    if history_end_ts not in index:
        raise ValueError(f"History cutoff {history_end_ts} is absent from the forecasting frame.")

    expected = _expected_hourly_index(
        history_end_ts + pd.Timedelta(hours=1), test_end_ts, label="recursive"
    )
    actual = index[(index > history_end_ts) & (index <= test_end_ts)]
    if not actual.equals(expected):
        missing = expected.difference(actual)
        unexpected = actual.difference(expected)
        raise ValueError(
            "The recursive interval is not a complete hourly sequence "
            f"({len(missing)} missing, {len(unexpected)} unexpected timestamps)."
        )
    if test_start_ts not in actual:
        raise ValueError(f"Test start {test_start_ts} is absent from the recursive interval.")
    return actual


def cv_pairs_from_splits(index: pd.DatetimeIndex, splits_subset: List[Dict]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Turn validated rolling split dictionaries into train/validation indices."""
    if not isinstance(index, pd.DatetimeIndex) or not index.is_monotonic_increasing:
        raise ValueError("The forecasting frame must have a sorted DatetimeIndex.")
    if not index.is_unique:
        raise ValueError("The forecasting frame contains duplicate timestamps.")
    pairs = []
    for fold_no, s in enumerate(splits_subset):
        train_start = pd.Timestamp(s["train_start"])
        train_end = pd.Timestamp(s["train_end"])
        test_start = pd.Timestamp(s["test_start"])
        test_end = pd.Timestamp(s["test_end"])
        if not (train_start <= train_end < test_start <= test_end):
            raise ValueError(f"Invalid temporal ordering in split {fold_no}: {s}")
        missing_boundaries = [
            timestamp
            for timestamp in (train_start, train_end, test_start, test_end)
            if timestamp not in index
        ]
        if missing_boundaries:
            raise ValueError(
                f"Split {fold_no} boundaries are absent from the forecasting frame: "
                f"{missing_boundaries}"
            )
        train_mask = (index >= pd.Timestamp(s["train_start"])) & (index <= pd.Timestamp(s["train_end"]))
        valid_mask = (index >= pd.Timestamp(s["test_start"]))  & (index <= pd.Timestamp(s["test_end"]))
        train_idx, valid_idx = np.flatnonzero(train_mask), np.flatnonzero(valid_mask)
        if train_idx.size == 0 or valid_idx.size == 0:
            raise ValueError(
                f"Split {fold_no} has no matching {'training' if train_idx.size == 0 else 'validation'} rows."
            )
        expected_valid = _expected_hourly_index(test_start, test_end, label=f"split {fold_no} test")
        actual_valid = index[valid_idx]
        if not actual_valid.equals(expected_valid):
            raise ValueError(f"Split {fold_no} test interval is not a complete hourly sequence.")
        pairs.append((train_idx, valid_idx))
    return pairs


def validation_history_end(split: Mapping[str, Any], policy: str) -> pd.Timestamp:
    """Return the last target observation available at the validation origin."""
    policy = str(policy).lower()
    if policy == "through_test_start":
        return pd.Timestamp(split["test_start"]) - pd.Timedelta(hours=1)
    if policy == "through_train_end":
        return pd.Timestamp(split["train_end"])
    raise ValueError(
        "validation_history_policy must be 'through_test_start' or 'through_train_end'"
    )

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
    allow_any_lags: bool = True,
    history_end: Optional[pd.Timestamp] = None,
) -> Tuple[List[str], List[str], int]:
    """
    Returns (allowed_cols, recalc_cols, min_required_lag_hours).

    - 'drop': allow non-lag; allow only lags >= (gap + horizon); rollings are DROPPED regardless of keep_rolling_stats.
    - 'own':  allow all own-target lags and (optionally) rollings; mark short ones for rebuild.
              Cross-target: allow only safe long lags; cross rollings dropped.
    - 'mutual': treat own and cross short lags and rolling statistics as rebuildable.
    - Rolling filtering (pre-policy, except 'drop' which drops all rollings anyway):
        * if keep_rolling_stats=False: all rmean/rstd dropped;
        * else keep only rmean/rstd with win_h >= min_rolling_win.
    - If allow_any_lags=False: drop ALL lag_* features regardless of policy/length.
    """
    assert lag_policy in ("drop", "own", "mutual")
    observed_through = pd.Timestamp(history_end) if history_end is not None else pd.Timestamp(split["train_end"])
    min_req_h = max(1, _hours_between(observed_through, pd.Timestamp(split["test_end"])))

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
               history_end: pd.Timestamp,
               preds_map: Dict[str, Dict[pd.Timestamp, float]]) -> float:
    if owner not in y_map:
        raise KeyError(f"No target history is available for dynamic feature owner {owner!r}.")
    if ts > history_end:
        owner_predictions = preds_map.get(owner, {})
        if ts not in owner_predictions:
            raise RuntimeError(f"Missing recursive prediction for {owner} at {ts}.")
        value = float(owner_predictions[ts])
    else:
        value = float(y_map[owner].get(ts, np.nan))
    if not np.isfinite(value):
        raise ValueError(f"Non-finite target history for {owner} at {ts}.")
    return value

def _rolling(owner: str,
             t: pd.Timestamp,
             win_h: int,
             y_map: Dict[str, pd.Series],
             history_end: pd.Timestamp,
             preds_map: Dict[str, Dict[pd.Timestamp, float]],
             stat: str) -> float:
    vals = []
    for k in range(1, win_h + 1):
        src_ts = t - pd.Timedelta(hours=k)
        vals.append(_y_or_pred(owner, src_ts, y_map, history_end, preds_map))
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    if stat == "mean":
        return float(a.mean())
    return float(a.std(ddof=1)) if a.size >= 2 else np.nan


def _validate_dynamic_history(
    df: pd.DataFrame,
    recalc_columns: Sequence[str],
    lag_meta: Mapping[str, Mapping[str, Any]],
    *,
    history_end: pd.Timestamp,
) -> None:
    """Fail before tuning if recursive target inputs lack their initial history."""
    lookback_by_owner: Dict[str, int] = {}
    for column in recalc_columns:
        meta = lag_meta.get(column)
        if meta is None:
            raise KeyError(f"No lag metadata is available for recalculated feature {column!r}.")
        owner = str(meta["owner"])
        lookback = int(meta.get("lag_h", meta.get("win_h", 0)))
        if lookback < 1:
            raise ValueError(f"Invalid lookback for recalculated feature {column!r}: {lookback}")
        lookback_by_owner[owner] = max(lookback_by_owner.get(owner, 0), lookback)

    for owner, lookback in lookback_by_owner.items():
        if owner not in df.columns:
            raise KeyError(f"Dynamic feature owner {owner!r} is absent from the forecasting frame.")
        start = pd.Timestamp(history_end) - pd.Timedelta(hours=lookback - 1)
        expected = _expected_hourly_index(start, history_end, label=f"{owner} history")
        values = pd.to_numeric(df[owner].reindex(expected), errors="coerce").to_numpy(dtype=float)
        if values.size != len(expected) or not np.isfinite(values).all():
            raise ValueError(
                f"{owner} lacks a complete finite {lookback}-hour history through {history_end}."
            )

# -----------------------------------------------------------------------------------
# Categorical utilities and model-safe preprocessing
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
    Use isinstance(..., pd.CategoricalDtype) per pandas>=2.2 guidance.
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
    return X.reindex(columns=columns)

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


def fit_preprocessor(
    model_name: str,
    X: pd.DataFrame,
    categorical_cols: Sequence[str],
    *,
    xgb_native_categorical: bool,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Fit the fold-local dataframe transformation and retain prediction state."""
    model = str(model_name).upper()
    base = ensure_nonempty_features(X)
    transformed, cat_names, trained_cols, ohe_cols = prepare_X_for_model(
        model,
        base,
        list(categorical_cols),
        xgb_native_categorical=xgb_native_categorical,
    )
    category_levels = {
        c: list(transformed[c].cat.categories)
        for c in cat_names
        if c in transformed and isinstance(transformed[c].dtype, pd.CategoricalDtype)
    }
    medians: Dict[str, float] = {}
    if model == "GBR":
        median_series = transformed.median(numeric_only=True).reindex(transformed.columns).fillna(0.0)
        medians = {str(c): float(v) for c, v in median_series.items()}
        transformed = transformed.fillna(median_series).fillna(0.0)

    state: Dict[str, Any] = {
        "model_name": model,
        "categorical_cols": list(categorical_cols),
        "cat_names": list(cat_names),
        "trained_cols": list(trained_cols),
        "ohe_cols": list(ohe_cols),
        "category_levels": category_levels,
        "medians": medians,
        "xgb_native_categorical": bool(xgb_native_categorical),
    }
    return transformed, state


def transform_with_preprocessor(X: pd.DataFrame, state: Mapping[str, Any]) -> pd.DataFrame:
    """Apply training-fold preprocessing without learning from validation data."""
    model = str(state["model_name"]).upper()
    base = ensure_nonempty_features(X)
    transformed, _, _, _ = prepare_X_for_model(
        model,
        base,
        list(state.get("categorical_cols", [])),
        xgb_native_categorical=bool(state.get("xgb_native_categorical", True)),
    )
    transformed = freeze_categories(transformed, dict(state.get("category_levels", {})))
    transformed = align_to_columns(transformed, list(state["trained_cols"]))

    if model in {"HGBR", "GBR"} or (
        model == "XGB" and not bool(state.get("xgb_native_categorical", True))
    ):
        ohe_cols = [c for c in state.get("ohe_cols", []) if c in transformed.columns]
        if ohe_cols:
            transformed[ohe_cols] = transformed[ohe_cols].fillna(0.0)
    if model == "GBR":
        medians = pd.Series(dict(state.get("medians", {}))).reindex(transformed.columns).fillna(0.0)
        transformed = transformed.fillna(medians).fillna(0.0)
    return transformed

def ensure_named_df(
    X,
    *,
    columns_hint: list[str] | None = None,
    like_est: Any | None = None
) -> pd.DataFrame:
    """
    Return X as a pandas.DataFrame with explicit column names.
    Avoids sklearn>=1.6 + LGBM warning:
      'X does not have valid feature names, but LGBMRegressor was fitted with feature names'
    Naming priority:
      1) columns_hint (e.g., trained_cols you already track)
      2) like_est.feature_names_in_ (recorded at fit time)
      3) LightGBM booster_.feature_name() if available
      4) generic ['col_0', 'col_1', ...]
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
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    y_map: Dict[str, pd.Series],
    preprocessor_map: Dict[str, Mapping[str, Any]],
    history_end: Optional[pd.Timestamp] = None,
    train_end: Optional[pd.Timestamp] = None,
) -> Dict[str, Dict[pd.Timestamp, float]]:
    """Roll through the horizon from the last observed target timestamp.

    ``train_end`` remains a backward-compatible alias. Pass ``history_end`` to
    distinguish the observation cutoff from the model-fitting cutoff.
    """
    if history_end is None and train_end is None:
        raise ValueError("history_end is required for recursive prediction.")
    if history_end is not None and train_end is not None:
        if pd.Timestamp(history_end) != pd.Timestamp(train_end):
            raise ValueError("history_end and legacy train_end specify different timestamps.")
    observed_through = pd.Timestamp(history_end if history_end is not None else train_end)
    roll_idx = _recursive_index_from_history(
        df_features.index,
        history_end=observed_through,
        test_start=test_start,
        test_end=test_end,
    )
    preds = {t: {} for t in targets}

    for target in targets:
        missing_components = [
            name
            for name, mapping in (
                ("estimator", est_map),
                ("allowed features", allowed_map),
                ("recalculated features", recalc_map),
                ("preprocessor", preprocessor_map),
            )
            if target not in mapping
        ]
        if missing_components:
            raise KeyError(f"Missing {', '.join(missing_components)} for target {target!r}.")

    for t in roll_idx:
        rows: Dict[str, pd.DataFrame] = {}
        for tgt in targets:
            cols = allowed_map.get(tgt, [])
            row = df_features.loc[t, cols].copy() if cols else pd.Series(dtype=float)
            for c in recalc_map.get(tgt, []):
                meta = lag_meta[c]; owner = meta["owner"]; kind = meta["kind"]
                if kind == "lag":
                    src = t - pd.Timedelta(hours=meta["lag_h"])
                    row[c] = _y_or_pred(owner, src, y_map, observed_through, preds)
                elif kind == "rmean":
                    row[c] = _rolling(owner, t, meta["win_h"], y_map, observed_through, preds, "mean")
                elif kind == "rstd":
                    row[c] = _rolling(owner, t, meta["win_h"], y_map, observed_through, preds, "std")

            raw_row = pd.DataFrame([row], index=pd.DatetimeIndex([t]))
            rows[tgt] = transform_with_preprocessor(raw_row, preprocessor_map[tgt])

        for tgt in targets:
            trained_cols = list(preprocessor_map[tgt]["trained_cols"])
            Xrow = ensure_named_df(rows[tgt], columns_hint=trained_cols, like_est=est_map[tgt])
            prediction = float(est_map[tgt].predict(Xrow)[0])
            if not np.isfinite(prediction):
                raise ValueError(f"Estimator returned a non-finite prediction for {tgt} at {t}.")
            preds[tgt][t] = prediction
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


def _dataframe_fingerprint(df: pd.DataFrame, columns: Sequence[str]) -> str:
    """Hash the actual tuning values so stale feature caches cannot be reused."""
    hashed = pd.util.hash_pandas_object(df.loc[:, list(columns)], index=True, categorize=True)
    digest = hashlib.sha256()
    digest.update(hashed.to_numpy(dtype=np.uint64, copy=False).tobytes())
    digest.update("\x1f".join(map(str, columns)).encode("utf-8"))
    return digest.hexdigest()


def build_fold_feature_cache(
    *,
    df: pd.DataFrame,
    X_all: pd.DataFrame,
    y_all: Dict[str, np.ndarray],
    targets: Sequence[str],
    feature_cols: List[str],
    lag_meta: Dict[str, Dict],
    pairs: List[Tuple[np.ndarray, np.ndarray]],
    splits_use: List[Dict],
    lag_policy: str,
    feature_selector: str,
    k_features: Optional[int],
    sfs_direction: str,
    sfs_cv: int,
    sfs_base_estimator: str,
    use_gpu: bool,
    keep_rolling_stats: bool,
    min_rolling_win: int,
    allow_any_lags: bool,
    validation_history_policy: str,
    known_categorical: Optional[Sequence[str]],
    checkpoint_dir: Optional[Path],
    cache_signature: Optional[str],
) -> List[Dict[str, Any]]:
    """Compute feature selection once per fold/target and checkpoint each result."""
    signature_payload = {
        "script_version": SCRIPT_VERSION,
        "data": cache_signature or _dataframe_fingerprint(df, [*feature_cols, *targets]),
        "targets": list(targets),
        "features": feature_cols,
        "splits": [
            {
                key: split.get(key)
                for key in ("train_start", "train_end", "test_start", "test_end", "scheme", "_source_index")
            }
            for split in splits_use
        ],
        "lag_policy": lag_policy,
        "feature_selector": feature_selector,
        "k_features": k_features,
        "sfs_direction": sfs_direction,
        "sfs_cv": sfs_cv,
        "sfs_base_estimator": sfs_base_estimator,
        "keep_rolling_stats": keep_rolling_stats,
        "min_rolling_win": min_rolling_win,
        "allow_any_lags": allow_any_lags,
        "validation_history_policy": validation_history_policy,
        "known_categorical": list(known_categorical or []),
    }
    signature = _stable_hash(signature_payload)
    cache_path = Path(checkpoint_dir) / "feature_selection.json" if checkpoint_dir else None
    cache: Dict[str, Any] = {"signature": signature, "folds": {}}
    if cache_path and cache_path.exists():
        loaded = _read_json(cache_path)
        if loaded.get("signature") != signature:
            raise RuntimeError(
                f"Incompatible feature cache at {cache_path}. Use a new checkpoint directory for changed inputs."
            )
        cache = loaded

    fold_specs: List[Dict[str, Any]] = []
    for fold_no, ((tr_idx, va_idx), split) in enumerate(zip(pairs, splits_use)):
        fold_key = str(fold_no)
        stored_fold = cache["folds"].setdefault(fold_key, {})
        target_specs: Dict[str, Any] = {}
        history_end = validation_history_end(split, validation_history_policy)
        _recursive_index_from_history(
            X_all.index,
            history_end=history_end,
            test_start=split["test_start"],
            test_end=split["test_end"],
        )
        _, fold_categorical = classify_features(
            X_all.iloc[tr_idx], [], known_categorical=known_categorical
        )
        for target in targets:
            if target not in y_all:
                raise KeyError(f"Missing target array for {target!r}.")
            if not np.isfinite(np.asarray(y_all[target][tr_idx], dtype=float)).all():
                raise ValueError(f"Split {fold_no} has non-finite {target} training targets.")
            if not np.isfinite(np.asarray(y_all[target][va_idx], dtype=float)).all():
                raise ValueError(f"Split {fold_no} has non-finite {target} validation targets.")
            if target not in stored_fold:
                allowed, recalc, min_required_lag = choose_features_for_split(
                    feature_cols,
                    lag_meta,
                    target=target,
                    other_targets=[other for other in targets if other != target],
                    split=split,
                    lag_policy=lag_policy,
                    keep_rolling_stats=keep_rolling_stats,
                    min_rolling_win=min_rolling_win,
                    allow_any_lags=allow_any_lags,
                    history_end=history_end,
                )
                selected = select_columns(
                    feature_selector,
                    X_all.iloc[tr_idx][allowed],
                    y_all[target][tr_idx],
                    k=k_features,
                    sfs_direction=sfs_direction,
                    sfs_cv=sfs_cv,
                    sfs_base_estimator=sfs_base_estimator,
                    sfs_use_gpu=use_gpu,
                )
                stored_fold[target] = {
                    "allowed": selected,
                    "recalc": [column for column in recalc if column in selected],
                    "categorical": [column for column in selected if column in fold_categorical],
                    "min_required_lag": int(min_required_lag),
                }
                if cache_path:
                    _atomic_write_json(cache_path, cache)
                print(
                    f"[feature cache] fold {fold_no + 1}/{len(pairs)} "
                    f"target={target} features={len(selected)}"
                )
            target_spec = dict(stored_fold[target])
            _validate_dynamic_history(
                df,
                target_spec.get("recalc", []),
                lag_meta,
                history_end=history_end,
            )
            target_specs[target] = target_spec
        fold_specs.append(
            {
                "fold_no": fold_no,
                "tr_idx": tr_idx,
                "va_idx": va_idx,
                "split": split,
                "history_end": history_end,
                "targets": target_specs,
            }
        )
    return fold_specs


def prepare_fold_matrices(
    *,
    X_all: pd.DataFrame,
    targets: Sequence[str],
    fold_specs: Sequence[Dict[str, Any]],
    model_name: str,
    lag_policy: str,
    xgb_native_categorical: bool,
) -> List[Dict[str, Any]]:
    """Build immutable fold matrices once instead of once per Optuna trial."""
    prepared_folds: List[Dict[str, Any]] = []
    for fold in fold_specs:
        tr_idx = fold["tr_idx"]
        va_idx = fold["va_idx"]
        target_data: Dict[str, Any] = {}
        for target in targets:
            spec = fold["targets"][target]
            Xtr, preprocessor = fit_preprocessor(
                model_name,
                X_all.iloc[tr_idx][spec["allowed"]],
                spec["categorical"],
                xgb_native_categorical=xgb_native_categorical,
            )
            Xva = None
            if lag_policy == "drop":
                Xva = transform_with_preprocessor(
                    X_all.iloc[va_idx][spec["allowed"]], preprocessor
                )
            target_data[target] = {
                **spec,
                "Xtr": Xtr,
                "Xva": Xva,
                "preprocessor": preprocessor,
            }
        prepared_folds.append({**fold, "targets": target_data})
    return prepared_folds


def _fit_fold_estimator(
    estimator: Any,
    *,
    model_name: str,
    Xtr: pd.DataFrame,
    ytr: np.ndarray,
    preprocessor: Mapping[str, Any],
    Xva: Optional[pd.DataFrame] = None,
    yva: Optional[np.ndarray] = None,
) -> None:
    """Fit one fold model with model-specific validation handling."""
    model = str(model_name).upper()
    has_validation = Xva is not None and yva is not None
    if model == "LGBM" and has_validation and lgb is not None:
        callbacks = [lgb.early_stopping(50, first_metric_only=True, verbose=False)]
        estimator.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="rmse", callbacks=callbacks)
    elif model == "XGB":
        estimator.set_params(eval_metric="rmse", verbosity=0)
        fit_kwargs: Dict[str, Any] = {"verbose": False}
        if has_validation:
            estimator.set_params(early_stopping_rounds=50)
            fit_kwargs["eval_set"] = [(Xva, yva)]
        estimator.fit(Xtr, ytr, **fit_kwargs)
    elif model == "CATBOOST":
        cat_indices = [Xtr.columns.get_loc(c) for c in preprocessor.get("cat_names", [])]
        fit_kwargs = {"cat_features": cat_indices, "verbose": False}
        if has_validation:
            fit_kwargs.update(eval_set=(Xva, yva), use_best_model=True, early_stopping_rounds=50)
        estimator.fit(Xtr, ytr, **fit_kwargs)
    else:
        estimator.fit(Xtr, ytr)


def _fit_objective_estimator(
    *,
    model_name: str,
    params: Mapping[str, Any],
    use_gpu: bool,
    Xtr: pd.DataFrame,
    ytr: np.ndarray,
    preprocessor: Mapping[str, Any],
    training_tail_early_stopping: bool,
) -> tuple[Any, Optional[int]]:
    """Fit an outer-fold estimator without using the outer validation targets."""
    model = str(model_name).upper()
    ytr = np.asarray(ytr)
    supports_external_es = model in {"LGBM", "XGB", "CATBOOST"}
    use_inner_es = training_tail_early_stopping and supports_external_es and len(Xtr) >= 72

    if use_inner_es:
        holdout_rows = min(max(24, int(math.ceil(0.10 * len(Xtr)))), len(Xtr) // 3)
        split_at = len(Xtr) - holdout_rows
        provisional = build_estimator(model_name, dict(params), use_gpu=use_gpu)
        _fit_fold_estimator(
            provisional,
            model_name=model_name,
            Xtr=Xtr.iloc[:split_at],
            ytr=ytr[:split_at],
            preprocessor=preprocessor,
            Xva=Xtr.iloc[split_at:],
            yva=ytr[split_at:],
        )
        effective = _effective_iteration_count(provisional, model_name)
        final_params = _set_iteration_parameter(params, model_name, effective)
        estimator = build_estimator(model_name, final_params, use_gpu=use_gpu)
        _fit_fold_estimator(
            estimator,
            model_name=model_name,
            Xtr=Xtr,
            ytr=ytr,
            preprocessor=preprocessor,
        )
        return estimator, effective

    estimator = build_estimator(model_name, dict(params), use_gpu=use_gpu)
    _fit_fold_estimator(
        estimator,
        model_name=model_name,
        Xtr=Xtr,
        ytr=ytr,
        preprocessor=preprocessor,
    )
    return estimator, _effective_iteration_count(estimator, model_name)


def _trial_parameter_key(params: Mapping[str, Any]) -> str:
    return _stable_hash(dict(params), length=20)


def _load_trial_progress(
    progress_dir: Optional[Path],
    params: Mapping[str, Any],
) -> tuple[Dict[str, Any], Optional[Path]]:
    key = _trial_parameter_key(params)
    path = Path(progress_dir) / f"trial_{key}.json" if progress_dir else None
    default = {"parameter_key": key, "params": dict(params), "folds": {}}
    if path and path.exists():
        loaded = _read_json(path)
        if loaded.get("parameter_key") != key:
            raise RuntimeError(f"Corrupt trial checkpoint: {path}")
        return loaded, path
    return default, path


def _save_trial_progress(path: Optional[Path], progress: Mapping[str, Any]) -> None:
    if path:
        _atomic_write_json(path, dict(progress))


def _aggregate_trial_metrics(
    fold_records: Sequence[Mapping[str, Any]], targets: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for target in targets:
        target_records = [record["metrics"][target] for record in fold_records if target in record["metrics"]]
        if target_records:
            result[target] = {
                metric: float(np.mean([float(record[metric]) for record in target_records]))
                for metric in ("RMSE", "MAPE%")
            }
    return result


def _effective_iteration_count(estimator: Any, model_name: str) -> Optional[int]:
    """Return the fitted tree/iteration count, including early-stopping results."""
    model = str(model_name).upper()
    try:
        if model == "LGBM":
            value = int(getattr(estimator, "best_iteration_", 0) or 0)
            return value if value > 0 else int(estimator.get_params()["n_estimators"])
        if model == "XGB":
            try:
                return int(estimator.best_iteration) + 1
            except (AttributeError, ValueError):
                return int(estimator.get_params()["n_estimators"])
        if model == "CATBOOST":
            value = int(getattr(estimator, "tree_count_", 0) or 0)
            return value if value > 0 else int(estimator.get_params()["iterations"])
        if model == "HGBR":
            return int(getattr(estimator, "n_iter_", estimator.get_params()["max_iter"]))
        if model == "GBR":
            return int(getattr(estimator, "n_estimators_", estimator.get_params()["n_estimators"]))
    except (KeyError, TypeError, ValueError):
        return None
    return None


def _aggregate_iteration_counts(
    fold_records: Sequence[Mapping[str, Any]], targets: Sequence[str]
) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for target in targets:
        values = [
            int(record["effective_iterations"][target])
            for record in fold_records
            if target in record.get("effective_iterations", {})
            and record["effective_iterations"][target] is not None
        ]
        if values:
            result[target] = max(1, int(round(float(np.median(values)))))
    return result


def _set_iteration_parameter(
    params: Mapping[str, Any], model_name: str, iterations: Optional[int]
) -> Dict[str, Any]:
    adjusted = dict(params)
    if iterations is None:
        return adjusted
    key = {
        "LGBM": "n_estimators",
        "XGB": "n_estimators",
        "CATBOOST": "iterations",
        "HGBR": "max_iter",
        "GBR": "n_estimators",
    }[str(model_name).upper()]
    adjusted[key] = max(1, int(iterations))
    return adjusted

# -----------------------------------------------------------------------------------
# Models & Optuna spaces
# -----------------------------------------------------------------------------------

def default_params(model_name: str) -> Dict[str, Any]:
    """Reasonable defaults when warm-starting peer models without a study."""
    if model_name == "LGBM":
        return dict(
            n_estimators=800, learning_rate=0.05, num_leaves=63, max_depth=-1,
            min_child_samples=40, subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
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
        return dict(learning_rate=0.05, max_iter=500, max_leaf_nodes=63, l2_regularization=0.0, random_state=RANDOM_STATE)
    if model_name == "GBR":
        return dict(n_estimators=500, learning_rate=0.05, max_depth=3, subsample=0.9, random_state=RANDOM_STATE)
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
            subsample_freq     = 1,
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
            learning_rate       = _f("learning_rate", 5e-3, 0.2, log=True),
            max_iter            = _i("max_iter", 100, 800, log=True),
            max_leaf_nodes      = _i("max_leaf_nodes", 15, 255),
            l2_regularization   = _f("l2_regularization", 1e-9, 10.0, log=True),
            random_state        = RANDOM_STATE,
        )
    if model_name == "GBR":
        return dict(
            n_estimators        = _i("n_estimators", 100, 1000, log=True),
            learning_rate       = _f("learning_rate", 5e-3, 0.2, log=True),
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
            est.set_params(
                device=("cuda" if use_gpu else "cpu"),
                tree_method="hist",
                enable_categorical=True,
            )
        except Exception:
            try:
                est.set_params(tree_method="hist")
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
# Cross-validation reporting with fold-fitted estimators
# -----------------------------------------------------------------------------------

def fresh_clone(est):
    """Create a fresh, unfitted estimator instance with the same hyperparameters."""
    cls = est.__class__
    if hasattr(est, "get_params"):
        return cls(**est.get_params(deep=True))
    return copy.deepcopy(est)

def _legacy_cv_scores_per_target_fixed(
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

    # Classify categorical predictors once for all folds.
    _, cat_cols_all = classify_features(df, targets, known_categorical=known_categorical)

    for (tr_idx, va_idx), split in zip(pairs, splits_use):
        tmp = {t: fresh_clone(est_map[t]) for t in targets}
        cat_cols_map: Dict[str, List[str]] = {}
        trained_cols_map: Dict[str, List[str]] = {}
        cat_idx_map: Dict[str, Optional[List[int]]] = {}
        cat_levels_map: Dict[str, Dict[str, List[str]]] = {}
        ohe_cols_map: Dict[str, List[str]] = {}

        # Fit one estimator per target on the current fold.
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
            # CatBoost consumes column indices; recursive preprocessing retains names.
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

def _legacy_make_objective_shared(
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

    # Pass early stopping to fit so fitted estimator state is not mutated.
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
            # Enable native categoricals and select the requested compute device.
            try:
                est.set_params(enable_categorical=xgb_native_categorical)
                if xgb_native_categorical:
                    if use_gpu:
                        # Retain compatibility with current and earlier XGBoost device APIs.
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

def _legacy_make_objective_single_target_given_peer(
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
            # Enable native categoricals and select the requested compute device.
            try:
                est.set_params(enable_categorical=xgb_native_categorical)
                if xgb_native_categorical:
                    if use_gpu:
                        # Retain compatibility with current and earlier XGBoost device APIs.
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

def _aggregate_rmse(
    metrics: Mapping[str, Mapping[str, float]],
    *,
    mode: str,
    weights: Mapping[str, float],
) -> float:
    values = {target: float(result["RMSE"]) for target, result in metrics.items()}
    if mode == "max":
        return max(values.values())
    if mode == "weighted":
        denominator = sum(float(weights.get(target, 1.0)) for target in values)
        numerator = sum(float(weights.get(target, 1.0)) * score for target, score in values.items())
        return numerator / max(denominator, 1e-12)
    if mode != "mean":
        raise ValueError("coupled_agg must be one of: mean, weighted, max")
    return float(np.mean(list(values.values())))


def _prediction_array(
    predictions: Mapping[pd.Timestamp, float], validation_index: pd.DatetimeIndex
) -> np.ndarray:
    missing = [timestamp for timestamp in validation_index if timestamp not in predictions]
    if missing:
        raise RuntimeError(f"Recursive prediction is missing {len(missing)} validation timestamps.")
    return np.asarray([predictions[timestamp] for timestamp in validation_index], dtype=float)


def _make_objective_shared(
    *,
    X_all: pd.DataFrame,
    y_all: Dict[str, np.ndarray],
    y_map: Dict[str, pd.Series],
    targets: Sequence[str],
    prepared_folds: Sequence[Dict[str, Any]],
    lag_meta: Dict[str, Dict],
    model_name: str,
    use_gpu: bool,
    lag_policy: str,
    seasonality: int,
    target_weights: Optional[Dict[str, float]],
    agg: str,
    progress_dir: Optional[Path],
):
    """Create a shared/single-target objective with fold-level restart support."""
    targets = list(targets)
    weights = target_weights or {target: 1.0 for target in targets}
    if progress_dir:
        Path(progress_dir).mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.trial.Trial) -> float:
        params = suggest_params(trial, model_name)
        progress, progress_path = _load_trial_progress(progress_dir, trial.params)
        fold_records: List[Dict[str, Any]] = []
        fold_scores: List[float] = []

        for fold_no, fold in enumerate(prepared_folds):
            fold_key = str(fold_no)
            record = progress["folds"].get(fold_key)
            if record is None:
                estimators: Dict[str, Any] = {}
                fold_effective_iterations: Dict[str, Optional[int]] = {}
                for target in targets:
                    target_data = fold["targets"][target]
                    estimator, effective_iterations = _fit_objective_estimator(
                        model_name=model_name,
                        params=params,
                        use_gpu=use_gpu,
                        Xtr=target_data["Xtr"],
                        ytr=y_all[target][fold["tr_idx"]],
                        preprocessor=target_data["preprocessor"],
                        training_tail_early_stopping=lag_policy == "drop",
                    )
                    estimators[target] = estimator
                    fold_effective_iterations[target] = effective_iterations

                metrics: Dict[str, Dict[str, float]] = {}
                if lag_policy == "drop":
                    for target in targets:
                        target_data = fold["targets"][target]
                        yhat = estimators[target].predict(target_data["Xva"])
                        measured = evaluate_predictions(
                            y_all[target][fold["va_idx"]], yhat, seasonality=seasonality
                        )
                        metrics[target] = {
                            "RMSE": float(measured["RMSE"]),
                            "MAPE%": float(measured["MAPE%"]),
                        }
                else:
                    allowed_map = {
                        target: list(fold["targets"][target]["allowed"]) for target in targets
                    }
                    recalc_map = {
                        target: list(fold["targets"][target]["recalc"]) for target in targets
                    }
                    preprocessor_map = {
                        target: fold["targets"][target]["preprocessor"] for target in targets
                    }
                    split = fold["split"]
                    recursive = roll_predict_multi(
                        estimators,
                        df_features=X_all,
                        targets=targets,
                        allowed_map=allowed_map,
                        recalc_map=recalc_map,
                        lag_meta=lag_meta,
                        history_end=pd.Timestamp(fold["history_end"]),
                        test_start=pd.Timestamp(split["test_start"]),
                        test_end=pd.Timestamp(split["test_end"]),
                        y_map=y_map,
                        preprocessor_map=preprocessor_map,
                    )
                    validation_index = X_all.index[fold["va_idx"]]
                    for target in targets:
                        yhat = _prediction_array(recursive[target], validation_index)
                        measured = evaluate_predictions(
                            y_all[target][fold["va_idx"]], yhat, seasonality=seasonality
                        )
                        metrics[target] = {
                            "RMSE": float(measured["RMSE"]),
                            "MAPE%": float(measured["MAPE%"]),
                        }

                record = {
                    "objective": _aggregate_rmse(metrics, mode=agg, weights=weights),
                    "metrics": metrics,
                    "effective_iterations": fold_effective_iterations,
                }
                progress["folds"][fold_key] = record
                progress["updated_at"] = datetime.now(timezone.utc).isoformat()
                _save_trial_progress(progress_path, progress)

            fold_records.append(record)
            fold_scores.append(float(record["objective"]))
            trial.report(float(np.mean(fold_scores)), step=fold_no)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        cv_metrics = _aggregate_trial_metrics(fold_records, targets)
        effective_iterations = _aggregate_iteration_counts(fold_records, targets)
        trial.set_user_attr("cv_metrics", cv_metrics)
        trial.set_user_attr("effective_iterations", effective_iterations)
        trial.set_user_attr("fold_count", len(fold_records))
        progress["complete"] = True
        progress["cv_metrics"] = cv_metrics
        progress["effective_iterations"] = effective_iterations
        progress["objective"] = float(np.mean(fold_scores))
        _save_trial_progress(progress_path, progress)
        return float(np.mean(fold_scores))

    return objective


def _load_or_fit_peer_models(
    *,
    prepared_folds: Sequence[Dict[str, Any]],
    y_all: Dict[str, np.ndarray],
    target: str,
    model_name: str,
    params: Dict[str, Any],
    use_gpu: bool,
    progress_dir: Optional[Path],
) -> Dict[int, Any]:
    """Fit fixed peer estimators once per fold and persist them between sessions."""
    models: Dict[int, Any] = {}
    peer_key = _stable_hash({"target": target, "params": params}, length=16)
    model_dir = Path(progress_dir) / f"peer_{_slug(target)}_{peer_key}" if progress_dir else None
    if model_dir:
        model_dir.mkdir(parents=True, exist_ok=True)

    for fold_no, fold in enumerate(prepared_folds):
        model_path = model_dir / f"fold_{fold_no:04d}.joblib" if model_dir else None
        estimator = None
        if model_path and model_path.exists():
            try:
                estimator = joblib.load(model_path)
            except Exception as exc:
                warnings.warn(f"Could not load peer checkpoint {model_path}: {exc}; refitting it.")
        if estimator is None:
            estimator = build_estimator(model_name, params, use_gpu=use_gpu)
            target_data = fold["targets"][target]
            _fit_fold_estimator(
                estimator,
                model_name=model_name,
                Xtr=target_data["Xtr"],
                ytr=y_all[target][fold["tr_idx"]],
                preprocessor=target_data["preprocessor"],
            )
            if model_path:
                _atomic_joblib_dump(model_path, estimator)
            print(f"[peer cache] target={target} fold={fold_no + 1}/{len(prepared_folds)}")
        models[fold_no] = estimator
    return models


def _make_objective_single_target_given_peer(
    *,
    X_all: pd.DataFrame,
    y_all: Dict[str, np.ndarray],
    y_map: Dict[str, pd.Series],
    targets: Sequence[str],
    t_main: str,
    peer_params: Dict[str, Any],
    prepared_folds: Sequence[Dict[str, Any]],
    lag_meta: Dict[str, Dict],
    model_name: str,
    use_gpu: bool,
    lag_policy: str,
    seasonality: int,
    progress_dir: Optional[Path],
):
    """Create an alternating mutual objective with cached fixed peer models."""
    if len(targets) != 2:
        raise ValueError("Alternating mutual optimization requires exactly two targets.")
    if lag_policy != "mutual":
        raise ValueError("Alternating peer optimization is only valid for lag_policy='mutual'.")
    t_peer = next(target for target in targets if target != t_main)
    if progress_dir:
        Path(progress_dir).mkdir(parents=True, exist_ok=True)
    peer_models = _load_or_fit_peer_models(
        prepared_folds=prepared_folds,
        y_all=y_all,
        target=t_peer,
        model_name=model_name,
        params=peer_params,
        use_gpu=use_gpu,
        progress_dir=progress_dir,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        params_main = suggest_params(trial, model_name)
        progress, progress_path = _load_trial_progress(progress_dir, trial.params)
        fold_records: List[Dict[str, Any]] = []
        fold_scores: List[float] = []

        for fold_no, fold in enumerate(prepared_folds):
            fold_key = str(fold_no)
            record = progress["folds"].get(fold_key)
            if record is None:
                main_estimator = build_estimator(model_name, params_main, use_gpu=use_gpu)
                main_data = fold["targets"][t_main]
                _fit_fold_estimator(
                    main_estimator,
                    model_name=model_name,
                    Xtr=main_data["Xtr"],
                    ytr=y_all[t_main][fold["tr_idx"]],
                    preprocessor=main_data["preprocessor"],
                )
                estimators = {t_main: main_estimator, t_peer: peer_models[fold_no]}
                allowed_map = {
                    target: list(fold["targets"][target]["allowed"]) for target in targets
                }
                recalc_map = {
                    target: list(fold["targets"][target]["recalc"]) for target in targets
                }
                preprocessor_map = {
                    target: fold["targets"][target]["preprocessor"] for target in targets
                }
                split = fold["split"]
                recursive = roll_predict_multi(
                    estimators,
                    df_features=X_all,
                    targets=targets,
                    allowed_map=allowed_map,
                    recalc_map=recalc_map,
                    lag_meta=lag_meta,
                    history_end=pd.Timestamp(fold["history_end"]),
                    test_start=pd.Timestamp(split["test_start"]),
                    test_end=pd.Timestamp(split["test_end"]),
                    y_map=y_map,
                    preprocessor_map=preprocessor_map,
                )
                validation_index = X_all.index[fold["va_idx"]]
                yhat = _prediction_array(recursive[t_main], validation_index)
                measured = evaluate_predictions(
                    y_all[t_main][fold["va_idx"]], yhat, seasonality=seasonality
                )
                metrics = {
                    t_main: {
                        "RMSE": float(measured["RMSE"]),
                        "MAPE%": float(measured["MAPE%"]),
                    }
                }
                record = {
                    "objective": metrics[t_main]["RMSE"],
                    "metrics": metrics,
                    "effective_iterations": {
                        t_main: _effective_iteration_count(main_estimator, model_name)
                    },
                }
                progress["folds"][fold_key] = record
                progress["updated_at"] = datetime.now(timezone.utc).isoformat()
                _save_trial_progress(progress_path, progress)

            fold_records.append(record)
            fold_scores.append(float(record["objective"]))
            trial.report(float(np.mean(fold_scores)), step=fold_no)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        cv_metrics = _aggregate_trial_metrics(fold_records, [t_main])
        effective_iterations = _aggregate_iteration_counts(fold_records, [t_main])
        trial.set_user_attr("cv_metrics", cv_metrics)
        trial.set_user_attr("effective_iterations", effective_iterations)
        trial.set_user_attr("fold_count", len(fold_records))
        progress["complete"] = True
        progress["cv_metrics"] = cv_metrics
        progress["effective_iterations"] = effective_iterations
        progress["objective"] = float(np.mean(fold_scores))
        _save_trial_progress(progress_path, progress)
        return float(np.mean(fold_scores))

    return objective


# -----------------------------------------------------------------------------------
# Top-level tuner orchestrator
# -----------------------------------------------------------------------------------


def make_checkpoint_storage(checkpoint_dir: Path, backend: str = "journal"):
    """Create persistent Optuna storage suitable for one Colab worker."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    backend = str(backend).lower()
    if backend == "journal":
        try:
            from optuna.storages import JournalStorage
            from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

            journal_path = str(checkpoint_dir / "optuna_journal.log")
            backend_object = JournalFileBackend(
                journal_path,
                lock_obj=JournalFileOpenLock(journal_path),
            )
            return JournalStorage(backend_object)
        except (ImportError, AttributeError):
            warnings.warn("Optuna JournalStorage is unavailable; falling back to SQLite.")
    if backend not in {"journal", "sqlite"}:
        raise ValueError("checkpoint storage backend must be 'journal' or 'sqlite'")
    database = (checkpoint_dir / "optuna.sqlite3").resolve()
    return f"sqlite:///{database.as_posix()}"


def _default_startup_trials(n_trials: int) -> int:
    return max(1, min(int(n_trials), min(10, max(3, int(n_trials) // 5))))


def _create_resumable_study(
    *,
    study_name: str,
    storage: Any,
    sampler_seed: int,
    n_startup_trials: int,
) -> optuna.Study:
    return optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=int(sampler_seed), n_startup_trials=int(n_startup_trials)
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=max(3, int(n_startup_trials)), n_warmup_steps=1
        ),
        storage=storage,
        study_name=study_name,
        load_if_exists=storage is not None,
    )


def _recover_interrupted_trials(study: optuna.Study) -> int:
    """Queue stale RUNNING parameter sets so fold checkpoints can finish them."""
    states = optuna.trial.TrialState
    trials = study.get_trials(deepcopy=False)
    completed_keys = {
        _trial_parameter_key(trial.params)
        for trial in trials
        if trial.state in (states.COMPLETE, states.PRUNED) and trial.params
    }
    waiting_keys = {
        _trial_parameter_key(trial.params)
        for trial in trials
        if trial.state == states.WAITING and trial.params
    }
    recovered = 0
    for trial in trials:
        if trial.state != states.RUNNING or not trial.params:
            continue
        key = _trial_parameter_key(trial.params)
        if key in completed_keys or key in waiting_keys:
            continue
        study.enqueue_trial(
            dict(trial.params),
            user_attrs={"recovered_from_trial": int(trial.number)},
        )
        waiting_keys.add(key)
        recovered += 1
    return recovered


def _optimize_to_total(
    study: optuna.Study,
    objective,
    *,
    total_trials: int,
) -> None:
    states = optuna.trial.TrialState
    recovered = _recover_interrupted_trials(study)
    finished = sum(
        trial.state in (states.COMPLETE, states.PRUNED)
        for trial in study.get_trials(deepcopy=False)
    )
    remaining = max(0, int(total_trials) - int(finished))
    print(
        f"[study] {study.study_name}: finished={finished}, remaining={remaining}, "
        f"recovered={recovered}"
    )
    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=1, gc_after_trial=True)
    complete = study.get_trials(deepcopy=False, states=(states.COMPLETE,))
    if not complete:
        raise RuntimeError(f"Study {study.study_name!r} has no completed trial.")

def _legacy_tune_multi_targets(
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
                    df=df, pairs=pairs, splits_use=splits_use, targets=[peer],
                    feature_cols=feature_cols, lag_meta=lag_meta, model_name=model_name, use_gpu=use_gpu,
                    lag_policy=lag_policy if lag_policy != "mutual" else "own",
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
            outer_loops_use = 1  # Independent tuning uses one pass.
        # Outer coordinate-descent style loops
        for outer in range(outer_loops_use):
            iter_tag = f"_iter{outer+1}" if separate_strategy == "alternating" else ""          
            for t_main in targets:
                t_peer = [x for x in targets if x != t_main]
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

    # Refit the selected configuration and assemble export metadata.
    last_split = splits_use[-1]
    allowed_map_final: Dict[str, List[str]] = {}
    recalc_map_final: Dict[str, List[str]] = {}

    last_train_mask = (df.index >= pd.Timestamp(last_split["train_start"])) & (df.index <= pd.Timestamp(last_split["train_end"]))
    tr_idx_final = np.flatnonzero(last_train_mask)

    # Reuse one categorical classification for both targets.
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

        # Apply the same preprocessing during the final refit.
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

    # Retain target-specific cross-validation scores for reporting.
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

def _final_refit_bounds(
    df: pd.DataFrame,
    splits: Sequence[Mapping[str, Any]],
    *,
    policy: str,
    explicit_end: Optional[Any],
) -> tuple[pd.Timestamp, pd.Timestamp]:
    policy = str(policy).lower()
    if explicit_end is not None:
        fit_end = pd.Timestamp(explicit_end)
    elif policy == "through_split_year_end":
        split_year = max(pd.Timestamp(split["test_end"]).year for split in splits)
        fit_end = pd.Timestamp(year=split_year, month=12, day=31, hour=23)
    elif policy == "through_last_test_end":
        fit_end = max(pd.Timestamp(split["test_end"]) for split in splits)
    elif policy in {"through_last_train_end", "last_train_window"}:
        fit_end = max(pd.Timestamp(split["train_end"]) for split in splits)
    else:
        raise ValueError(
            "final_refit_policy must be through_split_year_end, through_last_test_end, "
            "through_last_train_end, or last_train_window"
        )

    if policy == "last_train_window" and explicit_end is None:
        candidates = [split for split in splits if pd.Timestamp(split["train_end"]) == fit_end]
        fit_start = min(pd.Timestamp(split["train_start"]) for split in candidates)
    else:
        fit_start = min(pd.Timestamp(split["train_start"]) for split in splits)
    if fit_end > df.index.max():
        raise ValueError(f"Final refit end {fit_end} is after the last PQ timestamp {df.index.max()}.")
    if fit_start > fit_end:
        raise ValueError("Final refit range is empty.")
    return fit_start, fit_end


def tune_multi_targets(
    df: pd.DataFrame,
    splits: List[Dict],
    *,
    targets: Sequence[str] = ("P_Power", "Q_Power"),
    model_name: str = "LGBM",
    n_trials: int = 50,
    use_gpu: bool = True,
    max_splits: Optional[int] = None,
    split_selection: str = "stratified",
    seasonality: int = 24,
    storage: Optional[Any] = None,
    checkpoint_dir: Optional[os.PathLike | str] = None,
    checkpoint_storage_backend: str = "journal",
    cache_signature: Optional[str] = None,
    lag_policy: str = "drop",
    validation_history_policy: str = "through_test_start",
    share_hyperparams: bool = True,
    separate_strategy: str = "auto",
    outer_loops: int = 2,
    peer_warmstart_trials: int = 0,
    feature_selector: str = "mi_top_k",
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
    raw_target_columns: Optional[Sequence[str]] = None,
    xgb_native_categorical: bool = True,
    final_refit_policy: str = "through_split_year_end",
    final_refit_end: Optional[Any] = None,
    n_startup_trials: Optional[int] = None,
):
    """Tune P/Q specifications with resumable, leakage-safe rolling validation.

    The estimator is fitted only on each split's declared training rows.  By
    default, target observations through the hour before ``test_start`` are
    available to lag construction, and only the 24-hour test interval is
    recursively forecast.  Set ``validation_history_policy='through_train_end'``
    only when the entire gap is intentionally part of the forecast horizon.
    """
    canonical_names = {
        "LGBM": "LGBM",
        "XGB": "XGB",
        "CATBOOST": "CatBoost",
        "HGBR": "HGBR",
        "GBR": "GBR",
    }
    model_key = str(model_name).upper()
    if model_key not in canonical_names:
        raise ValueError(f"Unknown model: {model_name}")
    model_name = canonical_names[model_key]
    targets = [targets] if isinstance(targets, (str, np.str_)) else list(targets)
    if not targets:
        raise ValueError("At least one target is required.")
    missing = [target for target in targets if target not in df.columns]
    if missing:
        raise ValueError(f"Targets not found in dataframe: {missing}")
    if int(n_trials) < 1:
        raise ValueError("n_trials must be positive.")
    if lag_policy not in {"drop", "own", "mutual"}:
        raise ValueError("lag_policy must be drop, own, or mutual")
    if lag_policy == "mutual" and len(targets) < 2:
        raise ValueError("lag_policy='mutual' requires at least two jointly forecast targets.")
    validation_history_end(splits[0], validation_history_policy)

    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_root:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        if storage is None:
            storage = make_checkpoint_storage(checkpoint_root, checkpoint_storage_backend)

    discovered_raw_targets = set(targets)
    if raw_target_columns is None:
        discovered_raw_targets.update(
            column for column in ("P_Power", "Q_Power") if column in df.columns
        )
    else:
        discovered_raw_targets.update(map(str, raw_target_columns))
    unknown_raw_targets = [column for column in discovered_raw_targets if column not in df.columns]
    if unknown_raw_targets:
        raise ValueError(f"Raw target columns not found in dataframe: {unknown_raw_targets}")
    feature_cols = [column for column in df.columns if column not in discovered_raw_targets]
    X_all = df[feature_cols]
    y_map = {target: df[target] for target in discovered_raw_targets}
    y_all = {target: df[target].to_numpy() for target in targets}
    lag_meta = infer_lag_meta(feature_cols, sorted(discovered_raw_targets))
    splits_use = select_splits_for_tuning(splits, max_splits, strategy=split_selection)
    pairs = cv_pairs_from_splits(df.index, splits_use)
    fold_specs = build_fold_feature_cache(
        df=df,
        X_all=X_all,
        y_all=y_all,
        targets=targets,
        feature_cols=feature_cols,
        lag_meta=lag_meta,
        pairs=pairs,
        splits_use=splits_use,
        lag_policy=lag_policy,
        feature_selector=feature_selector,
        k_features=k_features,
        sfs_direction=sfs_direction,
        sfs_cv=sfs_cv,
        sfs_base_estimator=sfs_base_estimator,
        use_gpu=use_gpu,
        keep_rolling_stats=keep_rolling_stats,
        min_rolling_win=min_rolling_win,
        allow_any_lags=allow_any_lags,
        validation_history_policy=validation_history_policy,
        known_categorical=known_categorical,
        checkpoint_dir=checkpoint_root,
        cache_signature=cache_signature,
    )
    prepared_folds = prepare_fold_matrices(
        X_all=X_all,
        targets=targets,
        fold_specs=fold_specs,
        model_name=model_name,
        lag_policy=lag_policy,
        xgb_native_categorical=xgb_native_categorical,
    )

    startup_trials = int(n_startup_trials or _default_startup_trials(int(n_trials)))

    def study_name(base: str) -> str:
        prefix = f"{study_name_prefix}_" if study_name_prefix else ""
        return _slug(f"{prefix}{model_name}_{base}_{lag_policy}")

    def progress_dir(name: str) -> Optional[Path]:
        return checkpoint_root / "trials" / _slug(name) if checkpoint_root else None

    best_study_map: Dict[str, optuna.Study] = {}
    best_params_map: Dict[str, Dict[str, Any]] = {}

    if share_hyperparams:
        name = study_name(f"SHARED_{coupled_agg.upper()}")
        shared_study = _create_resumable_study(
            study_name=name,
            storage=storage,
            sampler_seed=_stable_seed(name),
            n_startup_trials=startup_trials,
        )
        objective = _make_objective_shared(
            X_all=X_all,
            y_all=y_all,
            y_map=y_map,
            targets=targets,
            prepared_folds=prepared_folds,
            lag_meta=lag_meta,
            model_name=model_name,
            use_gpu=use_gpu,
            lag_policy=lag_policy,
            seasonality=seasonality,
            target_weights=target_weights,
            agg=coupled_agg,
            progress_dir=progress_dir(name),
        )
        _optimize_to_total(shared_study, objective, total_trials=int(n_trials))
        shared_params = suggest_params(
            optuna.trial.FixedTrial(shared_study.best_trial.params), model_name
        )
        best_params_map = {target: dict(shared_params) for target in targets}
        best_study_map = {target: shared_study for target in targets}
        study = shared_study
        resolved_strategy = None
    else:
        resolved_strategy = str(separate_strategy).lower()
        if resolved_strategy == "auto":
            resolved_strategy = "alternating" if lag_policy == "mutual" else "independent"
        if resolved_strategy not in {"independent", "alternating"}:
            raise ValueError("separate_strategy must be auto, independent, or alternating")
        if lag_policy == "mutual" and resolved_strategy != "alternating":
            raise ValueError(
                "Unshared mutual models require separate_strategy='alternating'; "
                "independent optimization cannot construct peer forecasts."
            )
        if resolved_strategy == "alternating" and len(targets) != 2:
            raise ValueError("Alternating optimization currently requires exactly two targets.")

        best_params_map = {target: default_params(model_name) for target in targets}
        if peer_warmstart_trials:
            warm_policy = "own" if lag_policy == "mutual" else lag_policy
            for target in targets:
                warm_root = checkpoint_root / f"warm_{_slug(target)}" if checkpoint_root else None
                warm_specs = build_fold_feature_cache(
                    df=df,
                    X_all=X_all,
                    y_all={target: y_all[target]},
                    targets=[target],
                    feature_cols=feature_cols,
                    lag_meta=lag_meta,
                    pairs=pairs,
                    splits_use=splits_use,
                    lag_policy=warm_policy,
                    feature_selector=feature_selector,
                    k_features=k_features,
                    sfs_direction=sfs_direction,
                    sfs_cv=sfs_cv,
                    sfs_base_estimator=sfs_base_estimator,
                    use_gpu=use_gpu,
                    keep_rolling_stats=keep_rolling_stats,
                    min_rolling_win=min_rolling_win,
                    allow_any_lags=allow_any_lags,
                    validation_history_policy=validation_history_policy,
                    known_categorical=known_categorical,
                    checkpoint_dir=warm_root,
                    cache_signature=cache_signature,
                )
                warm_prepared = prepare_fold_matrices(
                    X_all=X_all,
                    targets=[target],
                    fold_specs=warm_specs,
                    model_name=model_name,
                    lag_policy=warm_policy,
                    xgb_native_categorical=xgb_native_categorical,
                )
                name = study_name(f"WARM_{target}")
                warm_study = _create_resumable_study(
                    study_name=name,
                    storage=storage,
                    sampler_seed=_stable_seed(name),
                    n_startup_trials=_default_startup_trials(int(peer_warmstart_trials)),
                )
                warm_objective = _make_objective_shared(
                    X_all=X_all,
                    y_all={target: y_all[target]},
                    y_map={target: y_map[target]},
                    targets=[target],
                    prepared_folds=warm_prepared,
                    lag_meta=lag_meta,
                    model_name=model_name,
                    use_gpu=use_gpu,
                    lag_policy=warm_policy,
                    seasonality=seasonality,
                    target_weights=None,
                    agg="mean",
                    progress_dir=progress_dir(name),
                )
                _optimize_to_total(
                    warm_study, warm_objective, total_trials=int(peer_warmstart_trials)
                )
                best_params_map[target] = suggest_params(
                    optuna.trial.FixedTrial(warm_study.best_trial.params), model_name
                )

        outer_count = max(1, int(outer_loops)) if resolved_strategy == "alternating" else 1
        for outer in range(outer_count):
            for target in targets:
                peer = next((other for other in targets if other != target), None)
                peer_key = (
                    _stable_hash(best_params_map[peer], length=10) if peer is not None else "none"
                )
                iteration = f"_iter{outer + 1}" if resolved_strategy == "alternating" else ""
                name = study_name(
                    f"{target}_SEPARATE_{resolved_strategy.upper()}{iteration}_peer{peer_key}"
                )
                target_study = _create_resumable_study(
                    study_name=name,
                    storage=storage,
                    sampler_seed=_stable_seed(name),
                    n_startup_trials=startup_trials,
                )
                if resolved_strategy == "independent":
                    objective = _make_objective_shared(
                        X_all=X_all,
                        y_all={target: y_all[target]},
                        y_map=y_map,
                        targets=[target],
                        prepared_folds=prepared_folds,
                        lag_meta=lag_meta,
                        model_name=model_name,
                        use_gpu=use_gpu,
                        lag_policy=lag_policy,
                        seasonality=seasonality,
                        target_weights=None,
                        agg="mean",
                        progress_dir=progress_dir(name),
                    )
                else:
                    assert peer is not None
                    objective = _make_objective_single_target_given_peer(
                        X_all=X_all,
                        y_all=y_all,
                        y_map=y_map,
                        targets=targets,
                        t_main=target,
                        peer_params=best_params_map[peer],
                        prepared_folds=prepared_folds,
                        lag_meta=lag_meta,
                        model_name=model_name,
                        use_gpu=use_gpu,
                        lag_policy=lag_policy,
                        seasonality=seasonality,
                        progress_dir=progress_dir(name),
                    )
                _optimize_to_total(target_study, objective, total_trials=int(n_trials))
                best_study_map[target] = target_study
                best_params_map[target] = suggest_params(
                    optuna.trial.FixedTrial(target_study.best_trial.params), model_name
                )
        study = best_study_map[targets[0]]

    fit_start, fit_end = _final_refit_bounds(
        df,
        splits,
        policy=final_refit_policy,
        explicit_end=final_refit_end,
    )
    final_mask = (df.index >= fit_start) & (df.index <= fit_end)
    final_split = {
        "train_start": fit_start,
        "train_end": fit_end,
        "test_start": fit_end + pd.Timedelta(hours=1),
        "test_end": fit_end + pd.Timedelta(hours=24),
    }
    final_feature_path = checkpoint_root / "final_feature_selection.json" if checkpoint_root else None
    final_signature = _stable_hash(
        {
            "cache_signature": cache_signature or _dataframe_fingerprint(df, [*feature_cols, *targets]),
            "fit_start": fit_start,
            "fit_end": fit_end,
            "lag_policy": lag_policy,
            "feature_selector": feature_selector,
            "k_features": k_features,
            "sfs_direction": sfs_direction,
            "sfs_cv": sfs_cv,
            "sfs_base_estimator": sfs_base_estimator,
            "keep_rolling_stats": keep_rolling_stats,
            "min_rolling_win": min_rolling_win,
            "allow_any_lags": allow_any_lags,
        }
    )
    final_cache: Dict[str, Any] = {"signature": final_signature, "targets": {}}
    if final_feature_path and final_feature_path.exists():
        loaded = _read_json(final_feature_path)
        if loaded.get("signature") != final_signature:
            raise RuntimeError(f"Incompatible final feature cache at {final_feature_path}")
        final_cache = loaded

    _, categorical_all = classify_features(
        X_all.loc[final_mask], [], known_categorical=known_categorical
    )
    validation_horizon_hours = sorted(
        {
            _hours_between(pd.Timestamp(split["test_start"]), pd.Timestamp(split["test_end"])) + 1
            for split in splits_use
        }
    )
    recursive_validation_steps = sorted(
        {
            _hours_between(
                validation_history_end(split, validation_history_policy),
                pd.Timestamp(split["test_end"]),
            )
            for split in splits_use
        }
    )
    estimators: Dict[str, Any] = {}
    allowed_map_final: Dict[str, List[str]] = {}
    metas: Dict[str, Any] = {}
    shared_final_iterations: Optional[int] = None
    if share_hyperparams:
        iteration_map = study.best_trial.user_attrs.get("effective_iterations", {})
        iteration_values = [int(value) for value in iteration_map.values() if value is not None]
        if iteration_values:
            shared_final_iterations = max(1, int(round(float(np.median(iteration_values)))))
    for target in targets:
        target_mask = final_mask & np.isfinite(df[target].to_numpy(dtype=float))
        target_idx = np.flatnonzero(target_mask)
        if target_idx.size == 0:
            raise ValueError(f"No finite final-refit rows for {target}.")
        cached_target = final_cache["targets"].get(target)
        if cached_target is None:
            allowed, recalc, _ = choose_features_for_split(
                feature_cols,
                lag_meta,
                target=target,
                other_targets=[other for other in targets if other != target],
                split=final_split,
                lag_policy=lag_policy,
                keep_rolling_stats=keep_rolling_stats,
                min_rolling_win=min_rolling_win,
                allow_any_lags=allow_any_lags,
                history_end=fit_end,
            )
            selected = select_columns(
                feature_selector,
                X_all.iloc[target_idx][allowed],
                y_all[target][target_idx],
                k=k_features,
                sfs_direction=sfs_direction,
                sfs_cv=sfs_cv,
                sfs_base_estimator=sfs_base_estimator,
                sfs_use_gpu=use_gpu,
            )
            cached_target = {
                "allowed": selected,
                "recalc": [column for column in recalc if column in selected],
                "categorical": [column for column in selected if column in categorical_all],
            }
            final_cache["targets"][target] = cached_target
            if final_feature_path:
                _atomic_write_json(final_feature_path, final_cache)

        Xfit, preprocessor = fit_preprocessor(
            model_name,
            X_all.iloc[target_idx][cached_target["allowed"]],
            cached_target["categorical"],
            xgb_native_categorical=xgb_native_categorical,
        )
        target_study = best_study_map[target]
        target_iteration_map = target_study.best_trial.user_attrs.get("effective_iterations", {})
        final_iterations = (
            shared_final_iterations
            if share_hyperparams
            else target_iteration_map.get(target)
        )
        final_params = _set_iteration_parameter(
            best_params_map[target], model_name, final_iterations
        )
        estimator = build_estimator(model_name, final_params, use_gpu=use_gpu)
        _fit_fold_estimator(
            estimator,
            model_name=model_name,
            Xtr=Xfit,
            ytr=y_all[target][target_idx],
            preprocessor=preprocessor,
        )
        estimators[target] = estimator
        allowed_map_final[target] = list(cached_target["allowed"])

        cv_metrics = target_study.best_trial.user_attrs.get("cv_metrics", {}).get(target, {})
        finished_trials = sum(
            trial.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
            for trial in target_study.get_trials(deepcopy=False)
        )
        metas[target] = {
            "target": target,
            "model": model_name,
            "raw_target_columns": sorted(discovered_raw_targets),
            "best_params": estimator.get_params() if hasattr(estimator, "get_params") else best_params_map[target],
            "tuned_params": best_params_map[target],
            "effective_iterations": int(final_iterations) if final_iterations is not None else None,
            "lag_policy": lag_policy,
            "features": list(cached_target["allowed"]),
            "recalc_features": list(cached_target["recalc"]),
            "preprocessing_state": preprocessor,
            "feature_selector": feature_selector,
            "k_features": int(k_features) if k_features is not None else None,
            "sfs_settings": {
                "direction": sfs_direction,
                "cv": int(sfs_cv),
                "base_estimator": sfs_base_estimator,
            } if str(feature_selector).startswith("sfs") else None,
            "rolling_settings": {
                "keep_rolling_stats": bool(keep_rolling_stats),
                "min_rolling_win": int(min_rolling_win),
            },
            "cv_rmse": float(cv_metrics.get("RMSE", math.nan)),
            "cv_mape%": float(cv_metrics.get("MAPE%", math.nan)),
            "cv_objective": float(target_study.best_value),
            "share_hyperparams": bool(share_hyperparams),
            "separate_strategy": resolved_strategy,
            "validation_history_policy": validation_history_policy,
            "validation_horizon_hours": validation_horizon_hours,
            "recursive_validation_steps": recursive_validation_steps,
            "future_covariate_assumption": (
                "Non-target feature values at forecast timestamps are treated as available at "
                "the forecast origin; operational weather inputs must therefore be forecast, not realized, values."
            ),
            "tuning_split_selection": split_selection,
            "tuning_split_count": len(splits_use),
            "tuning_split_source_indices": [int(split["_source_index"]) for split in splits_use],
            "n_trials_requested": int(n_trials),
            "n_trials_finished": int(finished_trials),
            "study_name": target_study.study_name,
            "final_refit_policy": final_refit_policy,
            "final_fit_start": fit_start,
            "final_fit_end": fit_end,
            "final_fit_rows": int(target_idx.size),
            "script_version": SCRIPT_VERSION,
        }
    return study, estimators, allowed_map_final, metas


# -----------------------------------------------------------------------------------
# Save / load
# -----------------------------------------------------------------------------------

def save_native(model, path, meta: dict | None = None):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    ext = ".joblib"
    if isinstance(model, (HistGradientBoostingRegressor, GradientBoostingRegressor)):
        _atomic_joblib_dump(p.with_suffix(".joblib"), model); ext = ".joblib"
    elif XGBRegressor is not None and isinstance(model, XGBRegressor):
        destination = p.with_suffix(".json")
        tmp = destination.with_name(f".{destination.stem}.{os.getpid()}.tmp.json")
        model.save_model(str(tmp)); os.replace(tmp, destination); ext = ".json"
    elif LGBMRegressor is not None and isinstance(model, LGBMRegressor):
        try:
            s = model.booster_.model_to_string()
            _atomic_write_bytes(p.with_suffix(".txt"), s.encode("utf-8")); ext = ".txt"
        except Exception:
            _atomic_joblib_dump(p.with_suffix(".joblib"), model); ext = ".joblib"
    elif CatBoostRegressor is not None and isinstance(model, CatBoostRegressor):
        destination = p.with_suffix(".cbm")
        tmp = destination.with_name(f".{destination.stem}.{os.getpid()}.tmp.cbm")
        model.save_model(str(tmp)); os.replace(tmp, destination); ext = ".cbm"
    else:
        _atomic_joblib_dump(p.with_suffix(".joblib"), model); ext = ".joblib"
    if meta:
        _atomic_write_json(p.with_suffix(".meta.json"), meta)
    return p.with_suffix(ext)

def load_native(path):
    p = Path(path); suf = p.suffix.lower()
    if suf == ".joblib": return joblib.load(p)
    if suf == ".json" and XGBRegressor is not None:
        m = XGBRegressor(); m.load_model(str(p)); return m
    if suf == ".txt" and lgb is not None:
        # model_str avoids LightGBM's Windows C-API limitation with non-ASCII paths.
        return lgb.Booster(model_str=p.read_bytes().decode("utf-8"))
    if suf == ".cbm" and CatBoostRegressor is not None:
        m = CatBoostRegressor(); m.load_model(str(p)); return m
    raise ValueError(f"Unsupported extension: {suf}")

# -----------------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------------

def _read_pq_xlsx(xlsx_path: str) -> List[pd.DataFrame]:
    dfs: List[pd.DataFrame] = []
    with pd.ExcelFile(xlsx_path) as workbook:
        for sheet in workbook.sheet_names:
            df = pd.read_excel(workbook, sheet_name=sheet)
            if "Datetime" not in df.columns:
                lower_to_original = {str(column).lower(): column for column in df.columns}
                for candidate in ("datetime", "date", "timestamp"):
                    if candidate in lower_to_original:
                        df.rename(columns={lower_to_original[candidate]: "Datetime"}, inplace=True)
                        break
            if "Datetime" not in df.columns:
                raise ValueError(f"PQ sheet {sheet!r} has no datetime column.")
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
    all_splits: List[List[Dict[str, str]]] = []
    with pd.ExcelFile(xlsx_path) as workbook:
        for sheet in workbook.sheet_names:
            df = pd.read_excel(workbook, sheet_name=sheet)
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
            keep = [c for c in df.columns if not str(c).startswith("unnamed")]
            sdicts = df[keep].to_dict("records")
            all_splits.append(sdicts)
    return all_splits

# -----------------------------------------------------------------------------------
# Resumable training grid and command-line interface
# -----------------------------------------------------------------------------------

DEFAULT_KNOWN_CATEGORICAL = [
    "Rainy", "hour", "weekday", "season_idx", "day_in_season", "season_len",
    "is_holiday", "is_day_before_hol", "is_day_after_hol", "is_weekend",
    "is_new_year", "is_jan2", "is_old_new_year", "is_orthxmas", "is_dec25",
]


def resolve_gpu_request(model_name: str, requested: bool) -> bool:
    """Test the requested learner once and fall back to CPU when GPU support is absent."""
    model = str(model_name).upper()
    if not requested or model in {"HGBR", "GBR"}:
        return False
    params = default_params("CatBoost" if model == "CATBOOST" else model)
    if model == "LGBM":
        params.update(n_estimators=5, num_leaves=7, min_child_samples=2)
    elif model == "XGB":
        params.update(n_estimators=5, max_depth=2)
    elif model == "CATBOOST":
        params.update(iterations=5, depth=2)
    rng = np.random.default_rng(RANDOM_STATE)
    X = pd.DataFrame(rng.normal(size=(32, 3)), columns=["a", "b", "c"])
    y = rng.normal(size=32)
    try:
        estimator = build_estimator(
            "CatBoost" if model == "CATBOOST" else model, params, use_gpu=True
        )
        if model == "CATBOOST":
            estimator.fit(X, y, verbose=False)
        else:
            estimator.fit(X, y)
        return True
    except Exception as exc:
        warnings.warn(
            f"{model_name} GPU preflight failed ({type(exc).__name__}: {exc}); using CPU."
        )
        return False


def derive_run_label(config: Mapping[str, Any]) -> str:
    explicit = config.get("run_label")
    if explicit:
        return str(explicit)
    selector = str(config.get("feature_selector", "all"))
    shared = bool(config.get("share_hyperparams", True))
    lag_policy = str(config.get("lag_policy", "mutual"))
    if selector.startswith("sfs"):
        return "SFS_shared" if shared else "SFS"
    if selector == "mi_top_k":
        return "mi_top_k_nolags" if lag_policy == "drop" else "mi_top_k"
    if lag_policy == "drop":
        return "LagDrop"
    if lag_policy == "own":
        return "own_lags"
    return "Shared_mutual_lags" if shared else "mutual_lags"


def run_training_grid(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Run model-sheet jobs, resuming partial jobs and skipping completed ones."""
    root = Path(config["project_root"]).expanduser().resolve()
    pq_path = root / config.get("pq_path", "Input/PQ.xlsx")
    splits_path = root / config.get("splits_path", "Input/splits.xlsx")
    output_dir = root / config.get("output_dir", "Models")
    checkpoint_dir = root / config.get("checkpoint_dir", "Forecasting intermediate")
    dry_run = bool(config.get("dry_run", False))
    if not pq_path.exists() or not splits_path.exists():
        raise FileNotFoundError(f"Missing input: PQ={pq_path.exists()} splits={splits_path.exists()}")

    with pd.ExcelFile(pq_path) as workbook:
        pq_sheets = list(workbook.sheet_names)
    with pd.ExcelFile(splits_path) as workbook:
        split_sheets = list(workbook.sheet_names)
    frames = _read_pq_xlsx(str(pq_path))
    splits_by_sheet = _read_splits_xlsx(str(splits_path))
    split_lookup = dict(zip(split_sheets, splits_by_sheet))
    missing_split_sheets = [name for name in pq_sheets if name not in split_lookup]
    extra_split_sheets = [name for name in split_sheets if name not in set(pq_sheets)]
    if missing_split_sheets or extra_split_sheets:
        raise ValueError(
            "PQ and split worksheets must match by name. "
            f"Missing split sheets={missing_split_sheets}; extra split sheets={extra_split_sheets}."
        )
    pair_count = len(frames)

    requested_sheets = config.get("sheet_indices")
    sheet_indices = list(range(pair_count)) if requested_sheets is None else [int(i) for i in requested_sheets]
    invalid = [index for index in sheet_indices if index < 0 or index >= pair_count]
    if invalid:
        raise ValueError(f"Invalid sheet indices: {invalid}; available indices are 0..{pair_count - 1}")
    models = list(config.get("models", ["LGBM"]))
    input_hashes = {
        "pq_sha256": _file_sha256(pq_path),
        "splits_sha256": _file_sha256(splits_path),
    }
    results: List[Dict[str, Any]] = []

    for requested_model in models:
        effective_gpu = resolve_gpu_request(requested_model, bool(config.get("use_gpu", True)))
        for sheet_index in sheet_indices:
            sheet_name = pq_sheets[sheet_index]
            sheet_splits = split_lookup[sheet_name]
            run_label = derive_run_label(config)
            job_config = {
                "script_version": SCRIPT_VERSION,
                **input_hashes,
                "pq_sheet_index": sheet_index,
                "pq_sheet_name": sheet_name,
                "split_sheet_name": sheet_name,
                "model": requested_model,
                "run_label": run_label,
                "targets": list(config.get("targets", ["P_Power", "Q_Power"])),
                "n_trials": int(config.get("n_trials", 30)),
                "use_gpu": effective_gpu,
                "max_splits": config.get("max_splits", 36),
                "split_selection": config.get("split_selection", "stratified"),
                "lag_policy": config.get("lag_policy", "mutual"),
                "validation_history_policy": config.get(
                    "validation_history_policy", "through_test_start"
                ),
                "share_hyperparams": bool(config.get("share_hyperparams", True)),
                "separate_strategy": config.get("separate_strategy", "auto"),
                "outer_loops": int(config.get("outer_loops", 2)),
                "peer_warmstart_trials": int(config.get("peer_warmstart_trials", 0)),
                "feature_selector": config.get("feature_selector", "all"),
                "k_features": config.get("k_features"),
                "sfs_direction": config.get("sfs_direction", "forward"),
                "sfs_cv": int(config.get("sfs_cv", 3)),
                "sfs_base_estimator": config.get("sfs_base_estimator", "ridge"),
                "keep_rolling_stats": bool(config.get("keep_rolling_stats", True)),
                "min_rolling_win": int(config.get("min_rolling_win", 2)),
                "allow_any_lags": bool(config.get("allow_any_lags", True)),
                "coupled_agg": config.get("coupled_agg", "mean"),
                "xgb_native_categorical": bool(config.get("xgb_native_categorical", True)),
                "final_refit_policy": config.get("final_refit_policy", "through_split_year_end"),
                "final_refit_end": config.get("final_refit_end"),
                "checkpoint_storage_backend": config.get("checkpoint_storage_backend", "journal"),
            }
            fingerprint = _stable_hash(job_config)
            job_name = _slug(f"{run_label}_{requested_model}_sheet{sheet_index}_{fingerprint[:12]}")
            job_dir = checkpoint_dir / job_name
            completed_path = job_dir / "completed.json"
            state_path = job_dir / "state.json"

            completed = _read_json(completed_path)
            if completed and completed.get("job_fingerprint") == fingerprint:
                artifacts = [Path(path) for path in completed.get("artifacts", [])]
                if artifacts and all(path.exists() for path in artifacts):
                    print(f"[skip complete] {job_name}")
                    results.append(completed)
                    continue

            selected = select_splits_for_tuning(
                sheet_splits,
                job_config["max_splits"],
                strategy=job_config["split_selection"],
            )
            print(
                f"[job] {job_name}: sheet={sheet_name} model={requested_model} "
                f"tuning_folds={len(selected)}/{len(sheet_splits)}"
            )
            if dry_run:
                results.append({"job": job_name, "status": "dry_run", "config": job_config})
                continue

            job_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(job_dir / "job_config.json", {"fingerprint": fingerprint, **job_config})
            state = {
                "job": job_name,
                "job_fingerprint": fingerprint,
                "status": "running",
                "started_or_resumed_at": datetime.now(timezone.utc).isoformat(),
            }
            _atomic_write_json(state_path, state)
            try:
                study, fitted, _, metadata = tune_multi_targets(
                    df=frames[sheet_index],
                    splits=sheet_splits,
                    targets=job_config["targets"],
                    model_name=requested_model,
                    n_trials=job_config["n_trials"],
                    use_gpu=effective_gpu,
                    max_splits=job_config["max_splits"],
                    split_selection=job_config["split_selection"],
                    checkpoint_dir=job_dir,
                    checkpoint_storage_backend=job_config["checkpoint_storage_backend"],
                    cache_signature=f"{input_hashes['pq_sha256']}:{input_hashes['splits_sha256']}:{sheet_index}",
                    lag_policy=job_config["lag_policy"],
                    validation_history_policy=job_config["validation_history_policy"],
                    share_hyperparams=job_config["share_hyperparams"],
                    separate_strategy=job_config["separate_strategy"],
                    outer_loops=job_config["outer_loops"],
                    peer_warmstart_trials=job_config["peer_warmstart_trials"],
                    feature_selector=job_config["feature_selector"],
                    k_features=job_config["k_features"],
                    sfs_direction=job_config["sfs_direction"],
                    sfs_cv=job_config["sfs_cv"],
                    sfs_base_estimator=job_config["sfs_base_estimator"],
                    keep_rolling_stats=job_config["keep_rolling_stats"],
                    min_rolling_win=job_config["min_rolling_win"],
                    allow_any_lags=job_config["allow_any_lags"],
                    coupled_agg=job_config["coupled_agg"],
                    study_name_prefix=f"sheet{sheet_index}",
                    known_categorical=DEFAULT_KNOWN_CATEGORICAL,
                    xgb_native_categorical=job_config["xgb_native_categorical"],
                    final_refit_policy=job_config["final_refit_policy"],
                    final_refit_end=job_config["final_refit_end"],
                )
                artifacts: List[str] = []
                for target in job_config["targets"]:
                    base = output_dir / f"{run_label}_{requested_model}_sheet{sheet_index}_{target}_best"
                    meta_path = base.with_suffix(".meta.json")
                    if meta_path.exists():
                        prior = _read_json(meta_path, {})
                        if prior.get("job_fingerprint") != fingerprint:
                            raise FileExistsError(
                                f"Refusing to overwrite {meta_path} from another or legacy configuration. "
                                "Change --run-label."
                            )
                    metadata[target].update(
                        {
                            "job_fingerprint": fingerprint,
                            "pq_sha256": input_hashes["pq_sha256"],
                            "splits_sha256": input_hashes["splits_sha256"],
                            "pq_sheet_index": sheet_index,
                            "pq_sheet_name": sheet_name,
                        }
                    )
                    native_path = save_native(fitted[target], base, meta=metadata[target])
                    artifacts.extend([str(native_path.resolve()), str(meta_path.resolve())])
                    print(f"[saved] {native_path}")

                completed = {
                    "job": job_name,
                    "job_fingerprint": fingerprint,
                    "status": "complete",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "artifacts": artifacts,
                }
                _atomic_write_json(completed_path, completed)
                _atomic_write_json(state_path, completed)
                results.append(completed)
            except KeyboardInterrupt:
                state.update(status="interrupted", updated_at=datetime.now(timezone.utc).isoformat())
                _atomic_write_json(state_path, state)
                print(f"[interrupted] {job_name}; rerun the same command to continue.")
                raise
            except Exception as exc:
                state.update(
                    status="failed",
                    updated_at=datetime.now(timezone.utc).isoformat(),
                    error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(),
                )
                _atomic_write_json(state_path, state)
                raise
    return results


def _build_cli_parser() -> argparse.ArgumentParser:
    bundle_dir = Path(__file__).resolve().parent
    bundled_project_root = bundle_dir.parent if (bundle_dir.parent / "Input").exists() else bundle_dir
    default_root = Path(os.environ.get("FORECAST_PROJECT_ROOT", bundled_project_root))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--models", nargs="+", default=["LGBM"], choices=["LGBM", "XGB", "CatBoost", "HGBR", "GBR"])
    parser.add_argument("--sheet-indices", nargs="*", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--max-splits", type=int, default=36)
    parser.add_argument("--all-splits", action="store_true")
    parser.add_argument("--split-selection", choices=["first", "even", "stratified"], default="stratified")
    parser.add_argument("--lag-policy", choices=["drop", "own", "mutual"], default="mutual")
    parser.add_argument(
        "--validation-history-policy",
        choices=["through_test_start", "through_train_end"],
        default="through_test_start",
    )
    parser.add_argument("--share-hyperparams", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--separate-strategy", choices=["auto", "independent", "alternating"], default="auto")
    parser.add_argument("--outer-loops", type=int, default=2)
    parser.add_argument("--peer-warmstart-trials", type=int, default=0)
    parser.add_argument("--feature-selector", choices=["all", "mi_top_k", "sfs"], default="all")
    parser.add_argument("--k-features", type=int, default=None)
    parser.add_argument("--sfs-direction", choices=["forward", "backward"], default="forward")
    parser.add_argument("--sfs-cv", type=int, default=3)
    parser.add_argument("--sfs-base-estimator", choices=["ridge", "hgb", "gbr", "lgbm", "xgb", "cat"], default="ridge")
    parser.add_argument("--keep-rolling-stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-rolling-win", type=int, default=2)
    parser.add_argument("--allow-any-lags", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--xgb-native-categorical", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--output-dir", default="Models")
    parser.add_argument("--checkpoint-dir", default="Forecasting intermediate")
    parser.add_argument("--checkpoint-storage-backend", choices=["journal", "sqlite"], default="journal")
    parser.add_argument(
        "--final-refit-policy",
        choices=["through_split_year_end", "through_last_test_end", "through_last_train_end", "last_train_window"],
        default="through_split_year_end",
    )
    parser.add_argument("--final-refit-end", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_cli_parser()
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    if not effective_argv:
        parser.print_help()
        return 0
    args = parser.parse_args(effective_argv)
    config = vars(args)
    config["max_splits"] = None if config.pop("all_splits") else config["max_splits"]
    run_training_grid(config)
    return 0


def run_from_main_settings(
    config_overrides: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Run the editable notebook configuration through the training grid."""

    # 1) Project paths. In Colab, replace PROJECT_ROOT with the mounted Drive path.
    bundle_dir = Path(__file__).resolve().parent
    default_project_root = bundle_dir.parent if (bundle_dir.parent / "Input").exists() else bundle_dir
    PROJECT_ROOT = Path(os.environ.get("FORECAST_PROJECT_ROOT", default_project_root)).expanduser().resolve()
    PQ_PATH = PROJECT_ROOT / "Input" / "PQ.xlsx"
    SPLITS_PATH = PROJECT_ROOT / "Input" / "splits.xlsx"
    OUTPUT_DIR = PROJECT_ROOT / "Models"
    CHECKPOINT_DIR = PROJECT_ROOT / "Forecasting intermediate"

    # 2) Models, worksheets, and tuning budget.
    MODELS_TO_TUNE = ["LGBM"]  # "LGBM", "XGB", "CatBoost", "HGBR", or "GBR"
    SHEET_INDICES = [0, 1, 2]  # Use None to select every PQ worksheet.
    N_TRIALS = 10
    ALL_SPLITS = True
    MAX_SPLITS = 36             # used only when ALL_SPLITS is False
    SPLIT_SELECTION = "stratified"  # "first", "even", or "stratified"

    # 3) Dynamic-feature and feature-selection settings.
    LAG_POLICY = "mutual"       # "drop", "own", or "mutual"
    FEATURE_SELECTOR = "all"    # "all", "mi_top_k", or "sfs"
    K_FEATURES = None            # number of features for mi_top_k/SFS; None keeps all
    SFS_DIRECTION = "forward"   # "forward" or "backward"
    SFS_CV = 3
    SFS_BASE_ESTIMATOR = "ridge"  # "ridge", "hgb", "gbr", "lgbm", "xgb", or "cat"
    KEEP_ROLLING_STATS = True
    MIN_ROLLING_WIN = 2
    ALLOW_ANY_LAGS = True

    # 4) Joint-versus-separate P/Q tuning.
    SHARE_HYPERPARAMS = False
    SEPARATE_STRATEGY = "alternating"  # "auto", "independent", or "alternating"
    OUTER_LOOPS = 2
    PEER_WARMSTART_TRIALS = 0
    COUPLED_AGG = "mean"

    # 5) Validation, refitting, runtime, and output settings.
    VALIDATION_HISTORY_POLICY = "through_test_start"
    FINAL_REFIT_POLICY = "through_split_year_end"
    FINAL_REFIT_END = None
    USE_GPU = True
    XGB_NATIVE_CATEGORICAL = True
    CHECKPOINT_STORAGE_BACKEND = "journal"  # "journal" or "sqlite"
    RUN_LABEL = "mutual_lags_separate"
    DRY_RUN = False

    config: Dict[str, Any] = {
        "project_root": PROJECT_ROOT,
        "pq_path": str(PQ_PATH),
        "splits_path": str(SPLITS_PATH),
        "output_dir": str(OUTPUT_DIR),
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "models": MODELS_TO_TUNE,
        "sheet_indices": SHEET_INDICES,
        "targets": ["P_Power", "Q_Power"],
        "n_trials": N_TRIALS,
        "max_splits": None if ALL_SPLITS else MAX_SPLITS,
        "split_selection": SPLIT_SELECTION,
        "lag_policy": LAG_POLICY,
        "validation_history_policy": VALIDATION_HISTORY_POLICY,
        "share_hyperparams": SHARE_HYPERPARAMS,
        "separate_strategy": SEPARATE_STRATEGY,
        "outer_loops": OUTER_LOOPS,
        "peer_warmstart_trials": PEER_WARMSTART_TRIALS,
        "feature_selector": FEATURE_SELECTOR,
        "k_features": K_FEATURES,
        "sfs_direction": SFS_DIRECTION,
        "sfs_cv": SFS_CV,
        "sfs_base_estimator": SFS_BASE_ESTIMATOR,
        "keep_rolling_stats": KEEP_ROLLING_STATS,
        "min_rolling_win": MIN_ROLLING_WIN,
        "allow_any_lags": ALLOW_ANY_LAGS,
        "coupled_agg": COUPLED_AGG,
        "use_gpu": USE_GPU,
        "xgb_native_categorical": XGB_NATIVE_CATEGORICAL,
        "run_label": RUN_LABEL,
        "checkpoint_storage_backend": CHECKPOINT_STORAGE_BACKEND,
        "final_refit_policy": FINAL_REFIT_POLICY,
        "final_refit_end": FINAL_REFIT_END,
        "dry_run": DRY_RUN,
    }
    if config_overrides:
        config.update(dict(config_overrides))
    return run_training_grid(config)


if __name__ == "__main__":
    raise SystemExit(main())

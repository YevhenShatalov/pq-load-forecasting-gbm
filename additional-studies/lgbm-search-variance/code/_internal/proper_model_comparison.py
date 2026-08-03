#!/usr/bin/env python3
"""Leakage-safe, calibrated comparison of the models stored in ``Models``.

This module is intentionally separate from Forecasting.py, Functions.py,
pq_model_compare.py, and dm_across_folder.py.  It reads their artifacts but
does not modify them.

Protocol
--------
1. Treat every ``*_best.meta.json`` pair as a fixed model specification.
   Native model files are not scored because Forecasting.py refits them on the
   complete PQ sheet.  For historical evaluation a fresh estimator is fitted
   at every daily forecast origin using only observations available then.
2. Generate leakage-safe recursive 24-hour point forecasts.  Future P and Q
   values are blanked before prediction, and all short lag/rolling features are
   validated and recomputed from observed plus previously forecast history.
3. Use a chronological calibration period to collect complete out-of-sample
   24-hour P/Q error trajectories.  The evaluation-period predictive
   distribution is formed from those trajectories, preserving dependence over
   the 24 horizons and between P and Q.
4. Retain RMSE, MAE, MAPE, SMAPE, WMAPE, MASE, pinball loss, CRPS, all Q05-Q95
   quantiles, and L05/U05-L95/U95 intervals.  Add coverage, width, interval
   score, horizon diagnostics, and Diebold-Mariano tests with Holm correction.

Default dates use October-December 2021 for rolling out-of-sample interval
calibration and January-February 2022 for evaluation, matching the original
project protocol.  Later 2022 observations are not used by default because the
full-scale war introduced a different structural regime.  The input PQ
workbook is accepted as prepared; this module does not repeat outlier detection
or SARIMA filling.

Example
-------
Audit the available LGBM pairs without fitting anything::

    python proper_model_comparison.py --audit-only --model-types LGBM

Run the complete default LGBM comparison::

    python proper_model_comparison.py --model-types LGBM

Run a smaller first check::

    python proper_model_comparison.py --model-types LGBM \
        --families mutual_lags own_lags --sheets 0 1
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import pq_model_compare as legacy

try:
    from scipy.stats import t as student_t
except Exception:  # pragma: no cover - scipy is available in the project runtime
    student_t = None


TARGETS = ("P_Power", "Q_Power")
TARGET_SHORT = {"P_Power": "P", "Q_Power": "Q"}
QUANTILE_GRID = tuple(np.round(np.arange(0.05, 1.00, 0.05), 2))
INTERVAL_COVERAGES = tuple(np.round(np.arange(0.05, 1.00, 0.05), 2))
KNOWN_CATEGORICAL = (
    "Rainy", "hour", "weekday", "season_idx", "day_in_season", "season_len",
    "is_holiday", "is_day_before_hol", "is_day_after_hol", "is_weekend",
    "is_new_year", "is_jan2", "is_old_new_year", "is_orthxmas", "is_dec25",
)
WEATHER_COLUMNS = {
    "Temperature", "DF_Point", "Relative_Humidity", "Precipitation",
    "Surface_Pressure", "Wind_Speed", "Wind_Speed_50", "UV_Index",
    "Irradiance", "PAR_Total", "Rainy",
}

_META_RE = re.compile(
    r"^(?:(?P<family>.+)_)?"
    r"(?P<model>LGBM|XGB|CatBoost|HGBR|GBR)_sheet(?P<sheet>\d+)_"
    r"(?P<target>P(?:_Power)?|Q(?:_Power)?)_best\.meta\.json$",
    re.IGNORECASE,
)
_LEGACY_META_RE = re.compile(
    r"^(?P<family>.+)_"
    r"(?P<model>LGBM|XGB|CatBoost|HGBR|GBR)_(?P<sheet>\d+)_"
    r"(?P<target>P(?:_Power)?|Q(?:_Power)?)_best\.meta\.json$",
    re.IGNORECASE,
)
_LAG_RE = re.compile(r"^(?P<target>P(?:_Power)?|Q(?:_Power)?)_lag_(?P<n>\d+)$", re.I)
_RMEAN_RE = re.compile(r"^(?P<target>P(?:_Power)?|Q(?:_Power)?)_rmean_(?P<n>\d+)$", re.I)
_RSTD_RE = re.compile(r"^(?P<target>P(?:_Power)?|Q(?:_Power)?)_rstd_(?P<n>\d+)$", re.I)


@dataclass(frozen=True)
class ArtifactSpec:
    family: str
    model_name: str
    sheet: int
    target: str
    lag_policy: str
    params: Dict[str, Any]
    features: Tuple[str, ...]
    recalc_features: Tuple[str, ...]
    metadata_path: Path


@dataclass(frozen=True)
class ModelPair:
    model_id: str
    family: str
    model_name: str
    sheet: int
    lag_policy: str
    p: ArtifactSpec
    q: ArtifactSpec


@dataclass
class PreparedEstimator:
    model_name: str
    estimator: Any
    feature_names: List[str]
    trained_columns: List[str]
    categorical: List[str]
    category_levels: Dict[str, List[Any]]
    cat_indices: List[int]
    one_hot: bool
    medians: Optional[pd.Series]
    xgb_native_categorical: bool


@dataclass(frozen=True)
class ComparisonConfig:
    data_path: str = "Input/PQ.xlsx"
    models_dir: str = "Models"
    output_dir: str = "Proper Model Comparison"
    calibration_start: str = "2021-10-01 00:00"
    calibration_end: str = "2021-12-31 23:00"
    evaluation_start: str = "2022-01-01 00:00"
    evaluation_end: str = "2022-02-28 23:00"
    model_types: Tuple[str, ...] = ("LGBM",)
    families: Optional[Tuple[str, ...]] = None
    sheets: Optional[Tuple[int, ...]] = None
    horizon: int = 24
    use_gpu: bool = False
    n_jobs: int = 8
    xgb_native_categorical: bool = True
    adaptive_calibration: bool = False
    stratify_weekend: bool = False
    minimum_pool_days: int = 12
    bootstrap_simulations: int = 0
    random_seed: int = 42
    include_unprefixed: bool = False
    include_baselines: bool = True
    assume_future_exogenous_available: bool = True
    max_models: Optional[int] = None


def _canonical_target(value: str) -> str:
    v = str(value).lower()
    if v in {"p", "p_power"}:
        return "P_Power"
    if v in {"q", "q_power"}:
        return "Q_Power"
    raise ValueError(f"Unknown target name: {value}")


def _clean_identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_model_pairs(
    models_dir: str | Path,
    *,
    model_types: Optional[Sequence[str]] = None,
    families: Optional[Sequence[str]] = None,
    sheets: Optional[Sequence[int]] = None,
    include_unprefixed: bool = False,
) -> Tuple[List[ModelPair], List[str]]:
    """Return complete P/Q metadata pairs and explicit skip reasons."""
    root = Path(models_dir)
    if not root.exists():
        raise FileNotFoundError(f"Models directory does not exist: {root}")

    model_filter = {m.upper() for m in model_types or ()}
    family_filter = {f.lower() for f in families or ()}
    sheet_filter = {int(s) for s in sheets or ()}
    grouped: Dict[Tuple[str, str, int], Dict[str, ArtifactSpec]] = {}
    skipped: List[str] = []

    for meta_path in sorted(root.glob("*_best.meta.json")):
        match = _META_RE.match(meta_path.name) or _LEGACY_META_RE.match(meta_path.name)
        if not match:
            skipped.append(f"unrecognized filename: {meta_path.name}")
            continue

        family_raw = match.group("family")
        family = family_raw or "unprefixed"
        model_name = match.group("model").upper()
        sheet = int(match.group("sheet"))
        target = _canonical_target(match.group("target"))

        if family == "unprefixed" and not include_unprefixed:
            skipped.append(f"unprefixed artifact excluded: {meta_path.name}")
            continue
        if model_filter and model_name not in model_filter:
            continue
        if family_filter and family.lower() not in family_filter:
            continue
        if sheet_filter and sheet not in sheet_filter:
            continue

        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            skipped.append(f"invalid JSON {meta_path.name}: {exc}")
            continue

        params = dict(metadata.get("best_params") or {})
        features = tuple(str(x) for x in (metadata.get("features") or ()))
        recalc = tuple(str(x) for x in (metadata.get("recalc_features") or ()))
        lag_policy = str(metadata.get("lag_policy") or "drop").lower()
        if not params:
            skipped.append(f"missing best_params: {meta_path.name}")
            continue
        if not features:
            skipped.append(f"missing features: {meta_path.name}")
            continue
        if lag_policy not in {"drop", "own", "mutual"}:
            skipped.append(f"unknown lag_policy={lag_policy}: {meta_path.name}")
            continue

        spec = ArtifactSpec(
            family=family,
            model_name=model_name,
            sheet=sheet,
            target=target,
            lag_policy=lag_policy,
            params=params,
            features=features,
            recalc_features=recalc,
            metadata_path=meta_path.resolve(),
        )
        grouped.setdefault((family, model_name, sheet), {})[target] = spec

    pairs: List[ModelPair] = []
    for (family, model_name, sheet), target_map in sorted(grouped.items()):
        if not all(t in target_map for t in TARGETS):
            skipped.append(f"incomplete P/Q pair: {family}|{model_name}|sheet{sheet}")
            continue
        p, q = target_map["P_Power"], target_map["Q_Power"]
        if p.lag_policy != q.lag_policy:
            skipped.append(
                f"different P/Q lag policies: {family}|{model_name}|sheet{sheet} "
                f"({p.lag_policy}, {q.lag_policy})"
            )
            continue
        model_id = _clean_identifier(f"{family}_{model_name}_sheet{sheet}")
        pairs.append(ModelPair(model_id, family, model_name, sheet, p.lag_policy, p, q))

    return pairs, skipped


def _parse_dynamic_feature(name: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    for kind, pattern in (("lag", _LAG_RE), ("rmean", _RMEAN_RE), ("rstd", _RSTD_RE)):
        match = pattern.match(name)
        if match:
            return kind, _canonical_target(match.group("target")), int(match.group("n"))
    return None, None, None


def _validate_pair_against_data(pair: ModelPair, df: pd.DataFrame, horizon: int) -> None:
    missing = sorted((set(pair.p.features) | set(pair.q.features)) - set(df.columns))
    if missing:
        raise ValueError(f"{pair.model_id}: missing sheet columns: {missing}")

    for artifact in (pair.p, pair.q):
        feature_set = set(artifact.features)
        recalc = set(artifact.recalc_features)
        extra_recalc = sorted(recalc - feature_set)
        if extra_recalc:
            raise ValueError(f"{pair.model_id}/{artifact.target}: recalc not in features: {extra_recalc}")
        if any(t in feature_set for t in TARGETS):
            raise ValueError(f"{pair.model_id}/{artifact.target}: raw target present as a feature")

        unsafe: List[str] = []
        for name in artifact.features:
            kind, _, n = _parse_dynamic_feature(name)
            if kind in {"rmean", "rstd"} and name not in recalc:
                unsafe.append(name)
            elif kind == "lag" and n is not None and n < horizon and name not in recalc:
                unsafe.append(name)
        if unsafe:
            raise ValueError(
                f"{pair.model_id}/{artifact.target}: dynamic features could expose future targets: {unsafe}"
            )


def _categorical_columns(frame: pd.DataFrame) -> List[str]:
    known = set(KNOWN_CATEGORICAL)
    result: List[str] = []
    for column in frame.columns:
        series = frame[column]
        if (
            column in known
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_bool_dtype(series.dtype)
        ):
            result.append(column)
        elif pd.api.types.is_integer_dtype(series.dtype):
            if series.nunique(dropna=True) <= max(24, int(0.02 * max(len(series), 1))):
                result.append(column)
    return result


def _sanitized_params(
    model_name: str,
    params: Mapping[str, Any],
    use_gpu: bool,
    n_jobs: int,
) -> Dict[str, Any]:
    clean = dict(params)
    model = model_name.upper()
    if model == "LGBM":
        clean["device"] = "gpu" if use_gpu else "cpu"
        clean["n_jobs"] = int(n_jobs)
    elif model == "XGB":
        clean["n_jobs"] = int(n_jobs)
    if model_name.upper() == "CATBOOST" and not use_gpu:
        clean.pop("devices", None)
        clean["task_type"] = "CPU"
        clean["thread_count"] = int(n_jobs)
    return clean


def _fit_estimator(
    model_name: str,
    params: Mapping[str, Any],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    use_gpu: bool,
    n_jobs: int,
    xgb_native_categorical: bool,
) -> PreparedEstimator:
    model = model_name.upper()
    original_columns = list(x_train.columns)
    categorical = _categorical_columns(x_train)
    x = x_train.copy()
    category_levels: Dict[str, List[Any]] = {}
    cat_indices: List[int] = []
    one_hot = model in {"HGBR", "GBR"} or (model == "XGB" and not xgb_native_categorical)
    medians: Optional[pd.Series] = None

    if model == "CATBOOST":
        for column in categorical:
            x[column] = x[column].astype("string").fillna("__nan__").astype(object)
        cat_indices = [x.columns.get_loc(c) for c in categorical]
    elif one_hot:
        for column in categorical:
            x[column] = x[column].astype("category")
        x = pd.get_dummies(x, columns=categorical, dummy_na=False)
        if model == "GBR":
            medians = x.median(numeric_only=True).reindex(x.columns).fillna(0.0)
            x = x.fillna(medians).fillna(0.0)
    else:
        for column in categorical:
            x[column] = x[column].astype("category")
            category_levels[column] = list(x[column].cat.categories)

    estimator = legacy.build_estimator_from_params(
        model_name,
        _sanitized_params(model_name, params, use_gpu, n_jobs),
        use_gpu=use_gpu,
    )
    if model == "CATBOOST":
        estimator.fit(x, y_train, cat_features=cat_indices, verbose=False)
    else:
        estimator.fit(x, y_train)

    return PreparedEstimator(
        model_name=model,
        estimator=estimator,
        feature_names=original_columns,
        trained_columns=list(x.columns),
        categorical=categorical,
        category_levels=category_levels,
        cat_indices=cat_indices,
        one_hot=one_hot,
        medians=medians,
        xgb_native_categorical=xgb_native_categorical,
    )


def _transform_for_prediction(prepared: PreparedEstimator, frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.reindex(columns=prepared.feature_names).copy()
    if prepared.model_name == "CATBOOST":
        for column in prepared.categorical:
            x[column] = x[column].astype("string").fillna("__nan__").astype(object)
    elif prepared.one_hot:
        for column in prepared.categorical:
            x[column] = x[column].astype("category")
        x = pd.get_dummies(x, columns=prepared.categorical, dummy_na=False)
        x = x.reindex(columns=prepared.trained_columns, fill_value=0)
        if prepared.model_name == "GBR":
            assert prepared.medians is not None
            x = x.fillna(prepared.medians).fillna(0.0)
    else:
        for column, levels in prepared.category_levels.items():
            x[column] = pd.Categorical(x[column], categories=levels)
        x = x.reindex(columns=prepared.trained_columns)
    return x


def _recompute_feature(
    timestamp: pd.Timestamp,
    name: str,
    history: Mapping[str, pd.Series],
) -> float:
    kind, target, n = _parse_dynamic_feature(name)
    if kind is None or target is None or n is None:
        raise ValueError(f"Not a recognized dynamic feature: {name}")
    series = history[target]
    if kind == "lag":
        return float(series.get(timestamp - pd.Timedelta(hours=n), np.nan))

    segment = series.loc[
        timestamp - pd.Timedelta(hours=n): timestamp - pd.Timedelta(hours=1)
    ]
    if segment.empty:
        return float("nan")
    if kind == "rmean":
        return float(segment.mean())
    # Functions.py builds rolling standard deviation with pandas' default ddof=1.
    return float(segment.std(ddof=1))


def _fit_pair_for_origin(
    df: pd.DataFrame,
    pair: ModelPair,
    train_end: pd.Timestamp,
    *,
    use_gpu: bool,
    n_jobs: int,
    xgb_native_categorical: bool,
) -> Dict[str, PreparedEstimator]:
    train_mask = df.index <= train_end
    if train_mask.sum() <= 24:
        raise ValueError(f"{pair.model_id}: not enough training rows through {train_end}")

    prepared: Dict[str, PreparedEstimator] = {}
    for artifact in (pair.p, pair.q):
        x_train = df.loc[train_mask, list(artifact.features)].copy()
        y_train = df.loc[train_mask, artifact.target].to_numpy(float)
        finite = np.isfinite(y_train)
        if not finite.all():
            x_train = x_train.loc[finite]
            y_train = y_train[finite]
        prepared[artifact.target] = _fit_estimator(
            artifact.model_name,
            artifact.params,
            x_train,
            y_train,
            use_gpu=use_gpu,
            n_jobs=n_jobs,
            xgb_native_categorical=xgb_native_categorical,
        )
    return prepared


def _forecast_day(
    df: pd.DataFrame,
    pair: ModelPair,
    estimators: Mapping[str, PreparedEstimator],
    test_index: pd.DatetimeIndex,
) -> Dict[str, np.ndarray]:
    if test_index.empty:
        raise ValueError("Empty daily forecast index")

    origin = test_index[0]
    # Future target values are explicitly unavailable, even though they exist in PQ.xlsx.
    history = {target: df[target].copy() for target in TARGETS}
    for target in TARGETS:
        history[target].loc[history[target].index >= origin] = np.nan

    output = {target: [] for target in TARGETS}
    artifacts = {"P_Power": pair.p, "Q_Power": pair.q}

    for timestamp in test_index:
        current: Dict[str, float] = {}
        for target in TARGETS:
            artifact = artifacts[target]
            row = df.loc[[timestamp], list(artifact.features)].copy()
            for feature in artifact.recalc_features:
                row.loc[timestamp, feature] = _recompute_feature(timestamp, feature, history)
            x = _transform_for_prediction(estimators[target], row)
            current[target] = float(estimators[target].estimator.predict(x)[0])

        # P and Q at the same timestamp are committed together.  Neither can see
        # the other's contemporaneous prediction or actual value.
        for target in TARGETS:
            output[target].append(current[target])
            history[target].loc[timestamp] = current[target]

    return {target: np.asarray(values, dtype=float) for target, values in output.items()}


def _complete_daily_origins(
    index: pd.DatetimeIndex,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    horizon: int,
) -> List[pd.DatetimeIndex]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    origins: List[pd.DatetimeIndex] = []
    for day in pd.date_range(start_ts.normalize(), end_ts.normalize(), freq="D"):
        test_index = pd.date_range(day, periods=horizon, freq="h")
        if test_index[-1] > end_ts:
            continue
        if test_index.isin(index).all():
            origins.append(test_index)
    return origins


def _run_point_period(
    df: pd.DataFrame,
    pair: ModelPair,
    *,
    start: str,
    end: str,
    horizon: int,
    use_gpu: bool,
    n_jobs: int,
    xgb_native_categorical: bool,
) -> Dict[str, pd.DataFrame]:
    rows: Dict[str, List[pd.DataFrame]] = {target: [] for target in TARGETS}
    origins = _complete_daily_origins(df.index, start, end, horizon)
    if not origins:
        raise ValueError(f"{pair.model_id}: no complete forecast days in {start} to {end}")

    for number, test_index in enumerate(origins, start=1):
        train_end = test_index[0] - pd.Timedelta(hours=1)
        estimators = _fit_pair_for_origin(
            df,
            pair,
            train_end,
            use_gpu=use_gpu,
            n_jobs=n_jobs,
            xgb_native_categorical=xgb_native_categorical,
        )
        forecasts = _forecast_day(df, pair, estimators, test_index)
        for target in TARGETS:
            actual = df.loc[test_index, target].to_numpy(float)
            frame = pd.DataFrame(
                {
                    "model_id": pair.model_id,
                    "target": target,
                    "ts": test_index,
                    "date": test_index[0].date(),
                    "horizon": np.arange(1, horizon + 1, dtype=int),
                    "train_end": train_end,
                    "y_true": actual,
                    "y_pred": forecasts[target],
                }
            )
            frame["error"] = frame["y_true"] - frame["y_pred"]
            rows[target].append(frame)
        print(
            f"[{pair.model_id}] {test_index[0].date()} "
            f"({number}/{len(origins)})",
            flush=True,
        )

    return {target: pd.concat(frames, ignore_index=True) for target, frames in rows.items()}


def _run_seasonal_naive_period(
    df: pd.DataFrame,
    *,
    lag: int,
    model_id: str,
    start: str,
    end: str,
    horizon: int,
) -> Dict[str, pd.DataFrame]:
    """Forecast every horizon from an observation available at the daily origin."""
    if lag < horizon:
        raise ValueError(f"Seasonal-naive lag {lag} is unsafe for a {horizon}-hour origin")
    rows: Dict[str, List[pd.DataFrame]] = {target: [] for target in TARGETS}
    origins = _complete_daily_origins(df.index, start, end, horizon)
    for test_index in origins:
        train_end = test_index[0] - pd.Timedelta(hours=1)
        source_index = test_index - pd.Timedelta(hours=lag)
        if not source_index.isin(df.index).all():
            continue
        for target in TARGETS:
            actual = df.loc[test_index, target].to_numpy(float)
            prediction = df.loc[source_index, target].to_numpy(float)
            frame = pd.DataFrame(
                {
                    "model_id": model_id,
                    "target": target,
                    "ts": test_index,
                    "date": test_index[0].date(),
                    "horizon": np.arange(1, horizon + 1, dtype=int),
                    "train_end": train_end,
                    "y_true": actual,
                    "y_pred": prediction,
                }
            )
            frame["error"] = frame["y_true"] - frame["y_pred"]
            rows[target].append(frame)
    if any(not frames for frames in rows.values()):
        raise ValueError(f"No complete seasonal-naive forecasts for lag {lag}, {start} to {end}")
    return {target: pd.concat(frames, ignore_index=True) for target, frames in rows.items()}


def _trajectory_cube(predictions: Mapping[str, pd.DataFrame], horizon: int) -> Tuple[List[Any], np.ndarray]:
    merged = predictions["P_Power"][["date", "horizon", "error"]].merge(
        predictions["Q_Power"][["date", "horizon", "error"]],
        on=["date", "horizon"],
        suffixes=("_P", "_Q"),
        validate="one_to_one",
    )
    dates: List[Any] = []
    trajectories: List[np.ndarray] = []
    for date, group in merged.groupby("date", sort=True):
        ordered = group.sort_values("horizon")
        if ordered["horizon"].tolist() != list(range(1, horizon + 1)):
            continue
        values = ordered[["error_P", "error_Q"]].to_numpy(float)
        if np.isfinite(values).all():
            dates.append(date)
            trajectories.append(values)
    if not trajectories:
        raise ValueError("No complete finite P/Q calibration error trajectories")
    return dates, np.stack(trajectories, axis=0)


def _np_quantile(values: np.ndarray, q: float, axis: int = 0, method: str = "linear") -> np.ndarray:
    try:
        return np.quantile(values, q, axis=axis, method=method)
    except TypeError:  # numpy < 1.22
        return np.quantile(values, q, axis=axis, interpolation=method)


def _signed_conformal_bounds(errors: np.ndarray, coverage: float) -> Tuple[np.ndarray, np.ndarray]:
    """Finite-sample equal-tailed order-statistic bounds by forecast horizon."""
    if errors.ndim != 2:
        raise ValueError("errors must have shape [calibration_days, horizon]")
    n = errors.shape[0]
    alpha = 1.0 - float(coverage)
    ordered = np.sort(errors, axis=0)
    lower_rank = int(math.floor((n + 1) * alpha / 2.0))
    upper_rank = int(math.ceil((n + 1) * (1.0 - alpha / 2.0)))
    lower_index = min(max(lower_rank, 1), n) - 1
    upper_index = min(max(upper_rank, 1), n) - 1
    return ordered[lower_index, :], ordered[upper_index, :]


def _crps_ensemble(y_true: np.ndarray, ensemble: np.ndarray) -> np.ndarray:
    y = np.asarray(y_true, dtype=float).ravel()
    sims = np.asarray(ensemble, dtype=float)
    term_one = np.mean(np.abs(sims - y[None, :]), axis=0)
    sorted_sims = np.sort(sims, axis=0)
    n = sorted_sims.shape[0]
    weights = 2 * np.arange(1, n + 1) - n - 1
    pair_term = np.sum(weights[:, None] * sorted_sims, axis=0) / (n * n)
    return term_one - pair_term


def _pinball(y_true: np.ndarray, quantile: np.ndarray, tau: float) -> np.ndarray:
    residual = np.asarray(y_true, dtype=float) - np.asarray(quantile, dtype=float)
    return np.maximum(tau * residual, (tau - 1.0) * residual)


def _add_probabilistic_predictions(
    evaluation: Mapping[str, pd.DataFrame],
    calibration: Mapping[str, pd.DataFrame],
    *,
    horizon: int,
    adaptive: bool,
    stratify_weekend: bool,
    minimum_pool_days: int,
    bootstrap_simulations: int,
    random_seed: int,
) -> Dict[str, pd.DataFrame]:
    calibration_dates, base_cube = _trajectory_cube(calibration, horizon)
    available_dates = list(calibration_dates)
    available_cube = base_cube.copy()
    result_parts: Dict[str, List[pd.DataFrame]] = {target: [] for target in TARGETS}
    rng = np.random.default_rng(random_seed)

    evaluation_by_target = {
        target: {date: g.sort_values("horizon").copy() for date, g in frame.groupby("date", sort=True)}
        for target, frame in evaluation.items()
    }
    common_dates = sorted(set(evaluation_by_target["P_Power"]) & set(evaluation_by_target["Q_Power"]))

    for date in common_dates:
        weekend = pd.Timestamp(date).weekday() >= 5
        indices = np.arange(len(available_dates))
        if stratify_weekend:
            same_type = np.array(
                [(pd.Timestamp(d).weekday() >= 5) == weekend for d in available_dates],
                dtype=bool,
            )
            if int(same_type.sum()) >= minimum_pool_days:
                indices = indices[same_type]
        pool = available_cube[indices]
        if pool.shape[0] < minimum_pool_days:
            raise ValueError(
                f"Only {pool.shape[0]} calibration trajectories for {date}; "
                f"minimum_pool_days={minimum_pool_days}"
            )
        if bootstrap_simulations > 0:
            sampled = rng.integers(0, pool.shape[0], size=bootstrap_simulations)
            ensemble_errors = pool[sampled]
        else:
            # Exact empirical ensemble: no Monte Carlo noise and no duplicated paths.
            ensemble_errors = pool

        actual_errors_for_update: List[np.ndarray] = []
        for target_index, target in enumerate(TARGETS):
            frame = evaluation_by_target[target][date]
            if frame["horizon"].tolist() != list(range(1, horizon + 1)):
                raise ValueError(f"Incomplete evaluation day for {target}: {date}")
            point = frame["y_pred"].to_numpy(float)
            actual = frame["y_true"].to_numpy(float)
            ensemble = point[None, :] + ensemble_errors[:, :, target_index]
            generated: Dict[str, Any] = {}

            for tau in QUANTILE_GRID:
                name = f"Q{int(round(tau * 100)):02d}"
                generated[name] = _np_quantile(ensemble, tau, axis=0, method="linear")
                generated[f"Pinball_{tau:.2f}"] = _pinball(actual, generated[name], tau)

            signed_errors = pool[:, :, target_index]
            for coverage in INTERVAL_COVERAGES:
                code = int(round(coverage * 100))
                lower_error, upper_error = _signed_conformal_bounds(signed_errors, coverage)
                lower = point + lower_error
                upper = point + upper_error
                generated[f"L{code:02d}"] = lower
                generated[f"U{code:02d}"] = upper
                generated[f"Covered{code:02d}"] = ((actual >= lower) & (actual <= upper)).astype(int)
                generated[f"Width{code:02d}"] = upper - lower
                alpha = 1.0 - coverage
                score = upper - lower
                score = score + (2.0 / alpha) * (lower - actual) * (actual < lower)
                score = score + (2.0 / alpha) * (actual - upper) * (actual > upper)
                generated[f"IntervalScore{code:02d}"] = score

            generated["CRPS"] = _crps_ensemble(actual, ensemble)
            pinball_columns = [f"Pinball_{tau:.2f}" for tau in QUANTILE_GRID]
            generated["Pinball_mean"] = np.column_stack(
                [generated[column] for column in pinball_columns]
            ).mean(axis=1)
            generated["CalibrationPoolDays"] = np.full(horizon, int(pool.shape[0]), dtype=int)
            frame = pd.concat(
                [frame.reset_index(drop=True), pd.DataFrame(generated)],
                axis=1,
            )
            result_parts[target].append(frame)
            actual_errors_for_update.append(frame["error"].to_numpy(float))

        if adaptive:
            trajectory = np.column_stack(actual_errors_for_update)[None, :, :]
            available_cube = np.concatenate([available_cube, trajectory], axis=0)
            available_dates.append(date)

    return {target: pd.concat(parts, ignore_index=True) for target, parts in result_parts.items()}


def _mase_scale(training_series: pd.Series, seasonal_period: int = 24) -> float:
    values = pd.to_numeric(training_series, errors="coerce").to_numpy(float)
    if values.size <= seasonal_period:
        return float("nan")
    differences = np.abs(values[seasonal_period:] - values[:-seasonal_period])
    differences = differences[np.isfinite(differences)]
    return float(np.mean(differences)) if differences.size else float("nan")


def _metric_row(frame: pd.DataFrame, mase_scale: float) -> Dict[str, float]:
    y = frame["y_true"].to_numpy(float)
    p = frame["y_pred"].to_numpy(float)
    error = y - p
    absolute = np.abs(error)
    finite = np.isfinite(y) & np.isfinite(p)
    y, p, error, absolute = y[finite], p[finite], error[finite], absolute[finite]
    nonzero = np.abs(y) > 1e-8
    denominator = np.abs(y) + np.abs(p)
    smape_mask = denominator > 1e-8

    result: Dict[str, float] = {
        "n": int(y.size),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(absolute)),
        "MAPE%": float(100.0 * np.mean(absolute[nonzero] / np.abs(y[nonzero]))) if nonzero.any() else np.nan,
        "SMAPE%": float(100.0 * np.mean(2.0 * absolute[smape_mask] / denominator[smape_mask])) if smape_mask.any() else np.nan,
        "WMAPE%": float(100.0 * np.sum(absolute) / np.sum(np.abs(y))) if np.sum(np.abs(y)) > 1e-8 else np.nan,
        "MASE": float(np.mean(absolute) / mase_scale) if np.isfinite(mase_scale) and mase_scale > 0 else np.nan,
        "Bias_y_minus_pred": float(np.mean(error)),
    }
    if "CRPS" in frame.columns:
        result["CRPS"] = float(frame.loc[finite, "CRPS"].mean())
    if "Pinball_mean" in frame.columns:
        result["Pinball_mean"] = float(frame.loc[finite, "Pinball_mean"].mean())
    for tau in QUANTILE_GRID:
        column = f"Pinball_{tau:.2f}"
        if column in frame.columns:
            result[column] = float(frame.loc[finite, column].mean())

    calibration_gaps: List[float] = []
    interval_scores: List[float] = []
    for coverage in INTERVAL_COVERAGES:
        code = int(round(coverage * 100))
        covered = f"Covered{code:02d}"
        width = f"Width{code:02d}"
        score = f"IntervalScore{code:02d}"
        if covered in frame.columns:
            empirical = float(frame.loc[finite, covered].mean())
            result[f"Coverage{code:02d}"] = empirical
            result[f"CoverageGap{code:02d}"] = empirical - coverage
            result[f"Width{code:02d}"] = float(frame.loc[finite, width].mean())
            result[f"IntervalScore{code:02d}"] = float(frame.loc[finite, score].mean())
            calibration_gaps.append(abs(empirical - coverage))
            interval_scores.append(result[f"IntervalScore{code:02d}"])
    if calibration_gaps:
        result["CalibrationMAE"] = float(np.mean(calibration_gaps))
        result["MeanIntervalScore"] = float(np.mean(interval_scores))
    return result


def _summary_tables(
    df: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame],
    period_start: str,
    *,
    model_id: str,
    period_name: str,
    mase_reference: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict[str, Any]] = []
    daily_rows: List[Dict[str, Any]] = []
    horizon_rows: List[Dict[str, Any]] = []
    start = pd.Timestamp(period_start)
    reference = mase_reference if mase_reference is not None else df

    for target in TARGETS:
        scale = _mase_scale(reference.loc[reference.index < start, target], seasonal_period=24)
        overall = _metric_row(predictions[target], scale)
        overall.update({"model_id": model_id, "period": period_name, "target": target})
        summary_rows.append(overall)
        for date, group in predictions[target].groupby("date", sort=True):
            row = _metric_row(group, scale)
            row.update({"model_id": model_id, "period": period_name, "target": target, "date": date})
            daily_rows.append(row)
        for horizon, group in predictions[target].groupby("horizon", sort=True):
            row = _metric_row(group, scale)
            row.update(
                {"model_id": model_id, "period": period_name, "target": target, "horizon": int(horizon)}
            )
            horizon_rows.append(row)

    return pd.DataFrame(summary_rows), pd.DataFrame(daily_rows), pd.DataFrame(horizon_rows)


def _loss_vector(frame: pd.DataFrame, loss: str) -> np.ndarray:
    key = loss.lower()
    error = frame["y_true"].to_numpy(float) - frame["y_pred"].to_numpy(float)
    if key == "squared":
        return error ** 2
    if key == "absolute":
        return np.abs(error)
    if key == "crps":
        return frame["CRPS"].to_numpy(float)
    if key == "pinball":
        return frame["Pinball_mean"].to_numpy(float)
    raise ValueError(f"Unknown DM loss: {loss}")


def _dm_test(loss_a: np.ndarray, loss_b: np.ndarray, horizon: int) -> Tuple[float, float]:
    differential = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    differential = differential[np.isfinite(differential)]
    n = differential.size
    if n <= max(5, horizon):
        return np.nan, np.nan
    centered = differential - differential.mean()
    gamma0 = float(np.dot(centered, centered) / n)
    long_run_variance = gamma0
    max_lag = min(horizon - 1, n - 1)
    for lag in range(1, max_lag + 1):
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
        long_run_variance += 2.0 * (1.0 - lag / horizon) * gamma
    if not np.isfinite(long_run_variance) or long_run_variance <= 0:
        return np.nan, np.nan

    statistic = float(differential.mean() / math.sqrt(long_run_variance / n))
    hln = math.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    statistic *= hln
    if student_t is not None:
        p_value = float(2.0 * student_t.sf(abs(statistic), df=n - 1))
    else:  # normal approximation fallback
        p_value = float(math.erfc(abs(statistic) / math.sqrt(2.0)))
    return statistic, p_value


def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if finite_indices.size == 0:
        return adjusted
    ordered = finite_indices[np.argsort(values[finite_indices])]
    m = ordered.size
    running = 0.0
    for rank, index in enumerate(ordered):
        candidate = min(1.0, (m - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _pairwise_dm_matrices(
    predictions: Mapping[str, pd.DataFrame],
    *,
    loss: str,
    horizon: int,
) -> Dict[str, pd.DataFrame]:
    ids = sorted(predictions)
    stat = pd.DataFrame(np.nan, index=ids, columns=ids, dtype=float)
    pval = stat.copy()
    sample = stat.copy()
    unique_pairs: List[Tuple[str, str]] = []
    unique_p: List[float] = []

    for model_id in ids:
        stat.loc[model_id, model_id] = 0.0
        pval.loc[model_id, model_id] = 1.0
        sample.loc[model_id, model_id] = len(predictions[model_id])

    for left, right in itertools.combinations(ids, 2):
        a = predictions[left][["ts", "y_true", "y_pred"]].copy()
        a["loss_a"] = _loss_vector(predictions[left], loss)
        b = predictions[right][["ts", "y_true", "y_pred"]].copy()
        b["loss_b"] = _loss_vector(predictions[right], loss)
        merged = a[["ts", "y_true", "loss_a"]].merge(
            b[["ts", "y_true", "loss_b"]], on="ts", suffixes=("_a", "_b"), validate="one_to_one"
        )
        if not np.allclose(merged["y_true_a"], merged["y_true_b"], equal_nan=True):
            raise ValueError(f"Actual values differ for DM pair {left}, {right}")
        s, p = _dm_test(merged["loss_a"].to_numpy(float), merged["loss_b"].to_numpy(float), horizon)
        n = len(merged)
        stat.loc[left, right], stat.loc[right, left] = s, -s if np.isfinite(s) else np.nan
        pval.loc[left, right] = pval.loc[right, left] = p
        sample.loc[left, right] = sample.loc[right, left] = n
        unique_pairs.append((left, right))
        unique_p.append(p)

    adjusted_values = _holm_adjust(unique_p)
    adjusted = pd.DataFrame(np.nan, index=ids, columns=ids, dtype=float)
    for model_id in ids:
        adjusted.loc[model_id, model_id] = 1.0
    for (left, right), p in zip(unique_pairs, adjusted_values):
        adjusted.loc[left, right] = adjusted.loc[right, left] = p
    return {"stat": stat, "p": pval, "p_holm": adjusted, "n": sample}


def _rank_summary(summary: pd.DataFrame) -> pd.DataFrame:
    ranked_parts: List[pd.DataFrame] = []
    for target, group in summary.groupby("target", sort=True):
        part = group.copy()
        for metric in ("RMSE", "MAE", "WMAPE%", "CRPS", "Pinball_mean", "CalibrationMAE"):
            if metric in part.columns:
                part[f"Rank_{metric}"] = part[metric].rank(method="min", ascending=True)
        objectives = [m for m in ("RMSE", "CRPS", "CalibrationMAE") if m in part.columns]
        pareto = np.ones(len(part), dtype=bool)
        values = part[objectives].to_numpy(float)
        for i in range(len(part)):
            for j in range(len(part)):
                if i == j:
                    continue
                if np.all(values[j] <= values[i]) and np.any(values[j] < values[i]):
                    pareto[i] = False
                    break
        part["Pareto_RMSE_CRPS_Calibration"] = pareto
        ranked_parts.append(part.sort_values("RMSE"))
    return pd.concat(ranked_parts, ignore_index=True)


def _legacy_prediction_columns() -> List[str]:
    columns = ["exp", "ts", "date", "y_true", "y_pred", "L80", "U80", "L95", "U95"]
    columns.extend(f"Q{int(round(tau * 100)):02d}" for tau in QUANTILE_GRID)
    for coverage in INTERVAL_COVERAGES:
        code = int(round(coverage * 100))
        if code not in {80, 95}:
            columns.extend([f"L{code:02d}", f"U{code:02d}"])
    columns.append("CRPS")
    columns.extend(f"Pinball_{tau:.2f}" for tau in QUANTILE_GRID)
    return columns


def _legacy_daily_columns() -> List[str]:
    columns = ["date", "exp", "n", "RMSE", "MAE", "MAPE%", "SMAPE%", "WMAPE%", "MASE", "CRPS"]
    columns.extend(f"Pinball_{tau:.2f}" for tau in QUANTILE_GRID)
    return columns


def _legacy_exp(model_name: str, lag_policy: str, sheet: int, target: str) -> str:
    return f"{model_name}|{lag_policy}|sheet{sheet}|{target}"


def _legacy_prediction_frame(frame: pd.DataFrame, exp: str) -> pd.DataFrame:
    output = frame.copy()
    output.insert(0, "exp", exp)
    required = _legacy_prediction_columns()
    missing = [column for column in required if column not in output.columns]
    if missing:
        raise ValueError(f"Cannot create legacy-compatible prediction sheet; missing: {missing}")
    return output[required]


def _legacy_daily_frame(frame: pd.DataFrame, exp: str) -> pd.DataFrame:
    output = frame.copy()
    output.insert(1, "exp", exp)
    required = _legacy_daily_columns()
    missing = [column for column in required if column not in output.columns]
    if missing:
        raise ValueError(f"Cannot create legacy-compatible daily sheet; missing: {missing}")
    return output[required]


def _legacy_pair_filename(pair: ModelPair) -> str:
    multi_sheet_families = {"mutual_lags", "own_lags", "shared_mutual_lags"}
    stem = f"{pair.family}_{pair.model_name}"
    if pair.family.lower() in multi_sheet_families or pair.sheet != 0:
        stem += f"_{pair.sheet}"
    return f"{stem}.xlsx"


def _write_pair_workbook(
    output_path: Path,
    evaluation_predictions: Mapping[str, pd.DataFrame],
    daily: pd.DataFrame,
    *,
    model_name: str,
    lag_policy: str,
    sheet: int,
) -> None:
    """Write the same four-sheet contract as pq_model_compare.py."""
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for target in TARGETS:
            exp = _legacy_exp(model_name, lag_policy, sheet, target)
            _legacy_prediction_frame(evaluation_predictions[target], exp).to_excel(
                writer, sheet_name=f"{target}_predictions", index=False
            )
            _legacy_daily_frame(daily[daily["target"] == target], exp).to_excel(
                writer, sheet_name=f"{target}_daily_metrics", index=False
            )


def _environment_manifest() -> Dict[str, Any]:
    packages: Dict[str, Optional[str]] = {}
    for name in ("numpy", "pandas", "scipy", "lightgbm", "xgboost", "catboost", "sklearn"):
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "unknown")
        except Exception:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def _new_run_directory(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    candidate = root / stem
    counter = 1
    while candidate.exists():
        candidate = root / f"{stem}_{counter}"
        counter += 1
    candidate.mkdir(parents=False)
    return candidate


def _read_pq(path: Path) -> Tuple[List[pd.DataFrame], List[str]]:
    excel = pd.ExcelFile(path)
    sheets = legacy.read_pq_xlsx(str(path))
    if len(sheets) != len(excel.sheet_names):
        raise RuntimeError("PQ sheet-name and loaded-sheet counts differ")
    return sheets, list(excel.sheet_names)


def audit_configuration(config: ComparisonConfig) -> Tuple[List[ModelPair], List[str], List[str]]:
    if config.n_jobs < 1:
        raise ValueError("n_jobs must be at least 1")
    pairs, skipped = discover_model_pairs(
        config.models_dir,
        model_types=config.model_types,
        families=config.families,
        sheets=config.sheets,
        include_unprefixed=config.include_unprefixed,
    )
    if config.max_models is not None:
        pairs = pairs[: int(config.max_models)]
    data_path = Path(config.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"PQ workbook does not exist: {data_path}")
    pq_sheets, sheet_names = _read_pq(data_path)

    cal_start = pd.Timestamp(config.calibration_start)
    cal_end = pd.Timestamp(config.calibration_end)
    eval_start = pd.Timestamp(config.evaluation_start)
    eval_end = pd.Timestamp(config.evaluation_end)
    if not (cal_start <= cal_end < eval_start <= eval_end):
        raise ValueError("Required order: calibration_start <= calibration_end < evaluation_start <= evaluation_end")

    valid: List[ModelPair] = []
    audit_notes: List[str] = []
    for pair in pairs:
        if pair.sheet >= len(pq_sheets):
            skipped.append(f"{pair.model_id}: sheet index {pair.sheet} is absent from PQ.xlsx")
            continue
        df = pq_sheets[pair.sheet]
        try:
            _validate_pair_against_data(pair, df, config.horizon)
            if cal_start < df.index.min() or eval_end > df.index.max():
                raise ValueError(
                    f"requested dates {cal_start}..{eval_end} outside {df.index.min()}..{df.index.max()}"
                )
            if not _complete_daily_origins(df.index, cal_start, cal_end, config.horizon):
                raise ValueError("no complete calibration days")
            if not _complete_daily_origins(df.index, eval_start, eval_end, config.horizon):
                raise ValueError("no complete evaluation days")
        except Exception as exc:
            skipped.append(f"{pair.model_id}: {exc}")
            continue
        weather_used = sorted((set(pair.p.features) | set(pair.q.features)) & WEATHER_COLUMNS)
        if weather_used:
            if not config.assume_future_exogenous_available:
                skipped.append(
                    f"{pair.model_id}: future exogenous availability not confirmed: "
                    f"{', '.join(weather_used)}"
                )
                continue
            audit_notes.append(
                f"{pair.model_id}: future exogenous weather columns used: {', '.join(weather_used)}"
            )
        valid.append(pair)

    if not valid:
        raise RuntimeError("No valid complete model pairs remain after auditing")
    return valid, skipped, [f"PQ sheet {i}: {name}" for i, name in enumerate(sheet_names)] + audit_notes


def compare_models_properly(config: ComparisonConfig = ComparisonConfig()) -> Path:
    """Run the full comparison and return the newly created output directory."""
    pairs, skipped, audit_notes = audit_configuration(config)
    data_path = Path(config.data_path).resolve()
    pq_sheets, sheet_names = _read_pq(data_path)

    package_keys = {
        "LGBM": "LGBM",
        "XGB": "XGB",
        "CATBOOST": "CatBoost",
        "HGBR": "HGBR",
        "GBR": "GBR",
    }
    unavailable = sorted(
        {
            pair.model_name
            for pair in pairs
            if legacy._pkg.get(package_keys[pair.model_name.upper()]) is None
        }
    )
    if unavailable:
        raise RuntimeError(
            "Estimator package(s) unavailable in this Python environment: "
            + ", ".join(unavailable)
            + ". Run with the same environment used to train those models."
        )
    run_dir = _new_run_directory(Path(config.output_dir).resolve())

    if config.assume_future_exogenous_available:
        warnings.warn(
            "Weather/exogenous values in evaluation rows are treated as available at forecast origin. "
            "If they are realized observations rather than forecast vintages, results are optimistic.",
            RuntimeWarning,
        )

    all_summary: List[pd.DataFrame] = []
    all_calibration_summary: List[pd.DataFrame] = []
    all_daily: List[pd.DataFrame] = []
    all_horizon: List[pd.DataFrame] = []
    predictions_for_dm: Dict[str, Dict[str, pd.DataFrame]] = {target: {} for target in TARGETS}

    for pair_number, pair in enumerate(pairs, start=1):
        print(f"\n=== {pair_number}/{len(pairs)}: {pair.model_id} ===", flush=True)
        df = pq_sheets[pair.sheet]
        calibration_point = _run_point_period(
            df,
            pair,
            start=config.calibration_start,
            end=config.calibration_end,
            horizon=config.horizon,
            use_gpu=config.use_gpu,
            n_jobs=config.n_jobs,
            xgb_native_categorical=config.xgb_native_categorical,
        )
        evaluation_point = _run_point_period(
            df,
            pair,
            start=config.evaluation_start,
            end=config.evaluation_end,
            horizon=config.horizon,
            use_gpu=config.use_gpu,
            n_jobs=config.n_jobs,
            xgb_native_categorical=config.xgb_native_categorical,
        )
        evaluation_full = _add_probabilistic_predictions(
            evaluation_point,
            calibration_point,
            horizon=config.horizon,
            adaptive=config.adaptive_calibration,
            stratify_weekend=config.stratify_weekend,
            minimum_pool_days=config.minimum_pool_days,
            bootstrap_simulations=config.bootstrap_simulations,
            random_seed=config.random_seed,
        )

        cal_summary, _, _ = _summary_tables(
            df,
            calibration_point,
            config.calibration_start,
            model_id=pair.model_id,
            period_name="calibration_point",
            mase_reference=pq_sheets[0],
        )
        summary, daily, horizon = _summary_tables(
            df,
            evaluation_full,
            config.evaluation_start,
            model_id=pair.model_id,
            period_name="evaluation",
            mase_reference=pq_sheets[0],
        )
        for table in (cal_summary, summary, daily, horizon):
            table.insert(1, "family", pair.family)
            table.insert(2, "model", pair.model_name)
            table.insert(3, "sheet", pair.sheet)
            table.insert(4, "lag_policy", pair.lag_policy)

        workbook_path = run_dir / _legacy_pair_filename(pair)
        _write_pair_workbook(
            workbook_path,
            evaluation_full,
            daily,
            model_name=pair.model_name,
            lag_policy=pair.lag_policy,
            sheet=pair.sheet,
        )
        all_calibration_summary.append(cal_summary)
        all_summary.append(summary)
        all_daily.append(daily)
        all_horizon.append(horizon)
        for target in TARGETS:
            predictions_for_dm[target][pair.model_id] = evaluation_full[target]

    if config.include_baselines:
        reference_df = pq_sheets[0]
        for lag in (24, 168):
            model_id = f"seasonal_naive_lag{lag}"
            print(f"\n=== baseline: {model_id} ===", flush=True)
            calibration_point = _run_seasonal_naive_period(
                reference_df,
                lag=lag,
                model_id=model_id,
                start=config.calibration_start,
                end=config.calibration_end,
                horizon=config.horizon,
            )
            evaluation_point = _run_seasonal_naive_period(
                reference_df,
                lag=lag,
                model_id=model_id,
                start=config.evaluation_start,
                end=config.evaluation_end,
                horizon=config.horizon,
            )
            evaluation_full = _add_probabilistic_predictions(
                evaluation_point,
                calibration_point,
                horizon=config.horizon,
                adaptive=config.adaptive_calibration,
                stratify_weekend=config.stratify_weekend,
                minimum_pool_days=config.minimum_pool_days,
                bootstrap_simulations=config.bootstrap_simulations,
                random_seed=config.random_seed,
            )
            cal_summary, _, _ = _summary_tables(
                reference_df,
                calibration_point,
                config.calibration_start,
                model_id=model_id,
                period_name="calibration_point",
                mase_reference=reference_df,
            )
            summary, daily, horizon = _summary_tables(
                reference_df,
                evaluation_full,
                config.evaluation_start,
                model_id=model_id,
                period_name="evaluation",
                mase_reference=reference_df,
            )
            for table in (cal_summary, summary, daily, horizon):
                table.insert(1, "family", "baseline")
                table.insert(2, "model", "SeasonalNaive")
                table.insert(3, "sheet", 0)
                table.insert(4, "lag_policy", f"lag{lag}")
            _write_pair_workbook(
                run_dir / f"{model_id}.xlsx",
                evaluation_full,
                daily,
                model_name="SeasonalNaive",
                lag_policy=f"lag{lag}",
                sheet=0,
            )
            all_calibration_summary.append(cal_summary)
            all_summary.append(summary)
            all_daily.append(daily)
            all_horizon.append(horizon)
            for target in TARGETS:
                predictions_for_dm[target][model_id] = evaluation_full[target]

    calibration_summary = pd.concat(all_calibration_summary, ignore_index=True)
    evaluation_summary = _rank_summary(pd.concat(all_summary, ignore_index=True))
    daily_metrics = pd.concat(all_daily, ignore_index=True)
    horizon_metrics = pd.concat(all_horizon, ignore_index=True)
    calibration_summary.to_csv(run_dir / "calibration_point_summary.csv", index=False)
    evaluation_summary.to_csv(run_dir / "evaluation_summary.csv", index=False)
    daily_metrics.to_csv(run_dir / "daily_metrics.csv", index=False)
    horizon_metrics.to_csv(run_dir / "horizon_metrics.csv", index=False)

    comparison_path = run_dir / "model_comparison.xlsx"
    with pd.ExcelWriter(comparison_path, engine="openpyxl") as writer:
        evaluation_summary.to_excel(writer, sheet_name="evaluation_summary", index=False)
        calibration_summary.to_excel(writer, sheet_name="calibration_summary", index=False)
        daily_metrics.to_excel(writer, sheet_name="daily_metrics", index=False)
        horizon_metrics.to_excel(writer, sheet_name="horizon_metrics", index=False)

        for target in TARGETS:
            short = TARGET_SHORT[target]
            for loss in ("squared", "absolute", "crps", "pinball"):
                matrices = _pairwise_dm_matrices(
                    predictions_for_dm[target], loss=loss, horizon=config.horizon
                )
                loss_code = {"squared": "SE", "absolute": "AE", "crps": "CRPS", "pinball": "PB"}[loss]
                matrices["stat"].to_excel(writer, sheet_name=f"{short}_{loss_code}_DM_stat")
                matrices["p"].to_excel(writer, sheet_name=f"{short}_{loss_code}_DM_p")
                matrices["p_holm"].to_excel(writer, sheet_name=f"{short}_{loss_code}_DM_holm")
                matrices["n"].to_excel(writer, sheet_name=f"{short}_{loss_code}_DM_n")

        protocol = pd.DataFrame(
            {
                "item": [
                    "calibration period",
                    "evaluation period",
                    "historical fitting",
                    "interval residual source",
                    "dependence",
                    "native model files",
                    "weather assumption",
                    "regime scope",
                    "DM interpretation",
                ],
                "value": [
                    f"{config.calibration_start} to {config.calibration_end}",
                    f"{config.evaluation_start} to {config.evaluation_end}",
                    "fresh daily refit through previous hour",
                    "out-of-sample recursive 24-hour calibration trajectories",
                    "whole paired P/Q daily error paths",
                    "not loaded for historical scoring",
                    "future exogenous values assumed available at forecast origin",
                    "post-February 2022 observations excluded by default because of wartime structural change",
                    "negative row-vs-column statistic means row model has lower loss",
                ],
            }
        )
        protocol.to_excel(writer, sheet_name="protocol", index=False)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": asdict(config),
        "data_path": str(data_path),
        "data_sha256": _file_sha256(data_path),
        "pq_sheets": sheet_names,
        "models": [
            {
                "model_id": pair.model_id,
                "family": pair.family,
                "model": pair.model_name,
                "sheet": pair.sheet,
                "lag_policy": pair.lag_policy,
                "p_metadata": str(pair.p.metadata_path),
                "p_metadata_sha256": _file_sha256(pair.p.metadata_path),
                "q_metadata": str(pair.q.metadata_path),
                "q_metadata_sha256": _file_sha256(pair.q.metadata_path),
            }
            for pair in pairs
        ],
        "skipped_artifacts": skipped,
        "audit_notes": audit_notes,
        "environment": _environment_manifest(),
        "methodological_notes": [
            "Prepared PQ.xlsx is read as supplied; preprocessing is not repeated.",
            "Native full-data model files are not used for historical scoring.",
            "October-December 2021 residuals are out-of-sample at each forecast origin but use metadata selected from 2021; they calibrate intervals and are not an independent point-model test.",
            "January-February 2022 is the default evaluation period; later 2022 data are excluded because of the wartime structural break.",
            "Calibration intervals use signed, horizon-specific finite-sample order statistics.",
            "CRPS and quantiles use the exact empirical trajectory ensemble unless bootstrap_simulations > 0.",
            "Whole P/Q residual trajectories preserve temporal and cross-target dependence.",
            "MASE uses one seasonal-naive scale from data preceding the evaluated period.",
            "DM uses hourly losses, horizon 24, Bartlett HAC, HLN correction, and Holm-adjusted p-values.",
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"\nComparison completed: {run_dir}", flush=True)
    return run_dir


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="Input/PQ.xlsx")
    parser.add_argument("--models-dir", default="Models")
    parser.add_argument("--output-dir", default="Proper Model Comparison")
    parser.add_argument("--calibration-start", default="2021-10-01 00:00")
    parser.add_argument("--calibration-end", default="2021-12-31 23:00")
    parser.add_argument("--evaluation-start", default="2022-01-01 00:00")
    parser.add_argument("--evaluation-end", default="2022-02-28 23:00")
    parser.add_argument("--model-types", nargs="+", default=["LGBM"])
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--sheets", nargs="+", type=int)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--adaptive-calibration", action="store_true")
    parser.add_argument("--stratify-weekend", action="store_true")
    parser.add_argument("--minimum-pool-days", type=int, default=12)
    parser.add_argument("--bootstrap-simulations", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--include-unprefixed", action="store_true")
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument(
        "--reject-future-exogenous",
        action="store_true",
        help="exclude models using weather values unless forecast-origin vintages are supplied",
    )
    parser.add_argument("--max-models", type=int)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    config = ComparisonConfig(
        data_path=args.data,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        calibration_start=args.calibration_start,
        calibration_end=args.calibration_end,
        evaluation_start=args.evaluation_start,
        evaluation_end=args.evaluation_end,
        model_types=tuple(args.model_types),
        families=tuple(args.families) if args.families else None,
        sheets=tuple(args.sheets) if args.sheets else None,
        horizon=args.horizon,
        use_gpu=args.gpu,
        n_jobs=args.n_jobs,
        adaptive_calibration=args.adaptive_calibration,
        stratify_weekend=args.stratify_weekend,
        minimum_pool_days=args.minimum_pool_days,
        bootstrap_simulations=args.bootstrap_simulations,
        random_seed=args.random_seed,
        include_unprefixed=args.include_unprefixed,
        include_baselines=not args.no_baselines,
        assume_future_exogenous_available=not args.reject_future_exogenous,
        max_models=args.max_models,
    )
    if args.audit_only:
        pairs, skipped, notes = audit_configuration(config)
        print(f"Valid model pairs: {len(pairs)}")
        for pair in pairs:
            print(
                f"  {pair.model_id}: family={pair.family}, model={pair.model_name}, "
                f"sheet={pair.sheet}, policy={pair.lag_policy}, "
                f"features(P/Q)={len(pair.p.features)}/{len(pair.q.features)}, "
                f"recalc(P/Q)={len(pair.p.recalc_features)}/{len(pair.q.recalc_features)}"
            )
        if skipped:
            print("Skipped artifacts/configurations:")
            for item in skipped:
                print(f"  - {item}")
        print("Audit notes:")
        for item in notes:
            print(f"  - {item}")
        return 0

    compare_models_properly(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

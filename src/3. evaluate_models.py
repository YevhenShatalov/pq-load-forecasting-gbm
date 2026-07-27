#!/usr/bin/env python3
"""Evaluate retained P-Q forecasting systems with rolling daily refits.

The evaluator reads prepared hourly inputs and paired model metadata without
modifying either source. It refits one P estimator and one Q estimator through
23:00 before each forecast day, recursively generates the next 24 hours, and
writes prediction workbooks, aggregate metrics, provenance records, and
publication-ready figures.

Protocol
--------
1. Treat every ``*_best.meta.json`` pair as a fixed model specification.
   Native model files are not scored because the training stage refits them on
   the complete PQ sheet. For historical evaluation a fresh estimator is
   fitted at every daily forecast origin using only observations available
   then.
2. Generate leakage-safe recursive 24-hour point forecasts.  Future P and Q
   values are blanked before prediction, and all short lag/rolling features are
   validated and recomputed from observed plus previously forecast history.
3. Use a chronological calibration period to collect complete out-of-sample
   24-hour P/Q error trajectories.  The evaluation-period predictive
   distribution is formed from those trajectories, preserving dependence over
   the 24 horizons and between P and Q.
4. Retain RMSE, MAE, MAPE, SMAPE, WMAPE, MASE, pinball loss, CRPS, all Q05-Q95
   quantiles, and L05/U05-L95/U95 intervals. Add coverage, width, interval
   score, horizon diagnostics, and the complete-system joint RMSE score.
5. Generate the all-origin trajectory, horizon-RMSE, and P-Q trade-off figures
   used in the manuscript. Formal daily-trajectory Diebold-Mariano inference is
   intentionally delegated to the separate statistical-comparison script.

Default dates use October-December 2021 for rolling out-of-sample interval
calibration and January-February 2022 for evaluation, matching the original
project protocol.  Later 2022 observations are not used by default because the
full-scale war introduced a different structural regime.  The input PQ
workbook is accepted as prepared; this module does not repeat outlier detection
or SARIMA filling.

Example
-------
Audit the available LGBM pairs without fitting anything::

    python "3. evaluate_models_proper.py" --audit-only --model-types LGBM

Run the complete default LGBM comparison::

    python "3. evaluate_models_proper.py" --model-types LGBM

Run a smaller first check::

    python "3. evaluate_models_proper.py" --model-types LGBM --families mutual_lags own_lags --sheets 0 1

Regenerate publication tables and figures from an existing completed run
without refitting any model::

    python "3. evaluate_models_proper.py" --figures-from "../Proper Model Comparison/LGBM Evaluated"
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

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # Plotting is optional for audit-only and evaluation-only use.
    plt = None

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
# Optional learner packages are loaded once and checked before evaluation.
_pkg: Dict[str, Any] = {}


def _try_import_model_packages() -> None:
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
        from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

        _pkg["HGBR"] = HistGradientBoostingRegressor
        _pkg["GBR"] = GradientBoostingRegressor
    except Exception:
        _pkg["HGBR"] = None
        _pkg["GBR"] = None


_try_import_model_packages()

BUNDLE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BUNDLE_DIR.parent if (BUNDLE_DIR.parent / "Input").exists() else Path.cwd()

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
    include_legacy_filenames: bool = False
    include_baselines: bool = True
    assume_future_exogenous_available: bool = True
    max_models: Optional[int] = None
    publication_outputs: bool = True
    publication_dir: Optional[str] = None
    figure_formats: Tuple[str, ...] = ("png", "pdf")
    figure_dpi: int = 600
    late_period_days: int = 5
    include_hourly_dm_diagnostics: bool = False


@dataclass(frozen=True)
class PublicationSystem:
    model_id: str
    short_label: str
    table_label: str
    family_label: str
    family: str
    model_name: str
    sheet: int
    lag_policy: str
    workbook: Path


def _canonical_target(value: str) -> str:
    v = str(value).lower()
    if v in {"p", "p_power"}:
        return "P_Power"
    if v in {"q", "q_power"}:
        return "Q_Power"
    raise ValueError(f"Unknown target name: {value}")


def _clean_identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _configure_console_errors() -> None:
    """Prevent non-ASCII project paths from crashing status output on Windows."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(errors="backslashreplace")
            except Exception:
                pass


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
    include_legacy_filenames: bool = False,
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
        match = _META_RE.match(meta_path.name)
        is_legacy_filename = False
        if match is None:
            match = _LEGACY_META_RE.match(meta_path.name)
            is_legacy_filename = match is not None
        if not match:
            skipped.append(f"unrecognized filename: {meta_path.name}")
            continue
        if is_legacy_filename and not include_legacy_filenames:
            skipped.append(f"legacy filename excluded: {meta_path.name}")
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
        try:
            metadata_target = _canonical_target(metadata.get("target", target))
        except ValueError as exc:
            skipped.append(f"{meta_path.name}: {exc}")
            continue
        if metadata_target != target:
            skipped.append(
                f"target mismatch between filename and metadata: {meta_path.name}"
            )
            continue
        metadata_model = str(metadata.get("model") or model_name).upper()
        if metadata_model != model_name:
            skipped.append(
                f"model mismatch between filename and metadata: {meta_path.name}"
            )
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
        key = (family, model_name, sheet)
        if target in grouped.setdefault(key, {}):
            skipped.append(
                f"duplicate metadata for {family}|{model_name}|sheet{sheet}|{target}: "
                f"{meta_path.name}"
            )
            continue
        grouped[key][target] = spec

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


def _artifact_training_signature(artifact: ArtifactSpec) -> str:
    params = dict(artifact.params)
    for runtime_key in (
        "device",
        "device_type",
        "devices",
        "n_jobs",
        "thread_count",
        "task_type",
    ):
        params.pop(runtime_key, None)
    payload = {
        "model": artifact.model_name.upper(),
        "sheet": int(artifact.sheet),
        "target": artifact.target,
        "lag_policy": artifact.lag_policy,
        "params": params,
        "features": list(artifact.features),
        "recalc_features": list(artifact.recalc_features),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _pair_training_signature(pair: ModelPair) -> Tuple[str, str]:
    return _artifact_training_signature(pair.p), _artifact_training_signature(pair.q)


def _pair_canonical_priority(pair: ModelPair) -> Tuple[int, str]:
    family = pair.family.lower()
    preferred = {"mutual_lags", "lagdrop"}
    known_aliases = {"shared_mutual_lags", "own"}
    if family in preferred:
        return 0, pair.model_id
    if family in known_aliases:
        return 2, pair.model_id
    return 1, pair.model_id


def _collapse_duplicate_model_pairs(
    pairs: Sequence[ModelPair],
) -> Tuple[List[ModelPair], List[str]]:
    """Count exact paired training specifications once before costly refitting."""
    grouped: Dict[Tuple[str, str], List[ModelPair]] = {}
    for pair in pairs:
        grouped.setdefault(_pair_training_signature(pair), []).append(pair)

    retained: List[ModelPair] = []
    notes: List[str] = []
    for members in grouped.values():
        ordered = sorted(members, key=_pair_canonical_priority)
        canonical = ordered[0]
        retained.append(canonical)
        for duplicate in ordered[1:]:
            notes.append(
                f"exact paired specification duplicate excluded: "
                f"{duplicate.model_id} -> {canonical.model_id}"
            )
    retained.sort(key=lambda pair: pair.model_id)
    return retained, notes


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
            kind, owner, n = _parse_dynamic_feature(name)
            if kind is not None and artifact.lag_policy == "drop":
                raise ValueError(
                    f"{pair.model_id}/{artifact.target}: target-derived feature {name} "
                    "is incompatible with lag_policy='drop'"
                )
            if (
                kind is not None
                and artifact.lag_policy == "own"
                and owner != artifact.target
            ):
                raise ValueError(
                    f"{pair.model_id}/{artifact.target}: cross-target feature {name} "
                    "is incompatible with lag_policy='own'"
                )
            if kind in {"rmean", "rstd"} and name not in recalc:
                unsafe.append(name)
            elif kind == "lag" and n is not None and n < horizon and name not in recalc:
                unsafe.append(name)
        unknown_recalc = sorted(
            name
            for name in artifact.recalc_features
            if _parse_dynamic_feature(name)[0] is None
        )
        if unknown_recalc:
            raise ValueError(
                f"{pair.model_id}/{artifact.target}: unrecognized recursive features: "
                f"{unknown_recalc}"
            )
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


def build_estimator_from_params(
    model_name: str,
    params: Mapping[str, Any],
    use_gpu: bool = True,
    xgb_native_categorical: bool = True,
) -> Any:
    """Construct a supported gradient estimator from stored hyperparameters."""
    name = (model_name or "").upper()
    if name == "LGBM" and _pkg["LGBM"] is not None:
        lgb = _pkg["LGBM"]
        model_params = dict(params or {})
        model_params.setdefault("verbose", -1)
        estimator = lgb.LGBMRegressor(**model_params)
        try:
            if use_gpu:
                for key in ("device", "device_type"):
                    try:
                        estimator.set_params(**{key: "gpu"})
                        break
                    except Exception:
                        pass
            else:
                for key in ("device", "device_type"):
                    try:
                        estimator.set_params(**{key: "cpu"})
                        break
                    except Exception:
                        pass
        except Exception:
            pass
        return estimator
    if name == "XGB" and _pkg["XGB"] is not None:
        xgb = _pkg["XGB"]
        estimator = xgb.XGBRegressor(**dict(params or {}))
        try:
            estimator.set_params(enable_categorical=bool(xgb_native_categorical))
        except Exception:
            pass
        try:
            estimator.set_params(device="cuda" if use_gpu else "cpu")
        except Exception:
            try:
                estimator.set_params(tree_method="gpu_hist" if use_gpu else "hist")
                if use_gpu:
                    estimator.set_params(predictor="gpu_predictor")
            except Exception:
                pass
        return estimator
    if name == "CATBOOST" and _pkg["CatBoost"] is not None:
        CatBoostRegressor = _pkg["CatBoost"]
        model_params = dict(params or {})
        model_params.setdefault("verbose", False)
        if use_gpu:
            model_params.setdefault("task_type", "GPU")
        return CatBoostRegressor(**model_params)
    if name == "HGBR" and _pkg["HGBR"] is not None:
        return _pkg["HGBR"](**dict(params or {}))
    if name == "GBR" and _pkg["GBR"] is not None:
        return _pkg["GBR"](**dict(params or {}))
    raise RuntimeError(f"Requested model '{model_name}' not available in this environment.")


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

    estimator = build_estimator_from_params(
        model_name,
        _sanitized_params(model_name, params, use_gpu, n_jobs),
        use_gpu=use_gpu,
        xgb_native_categorical=xgb_native_categorical,
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
            invalid = [
                feature
                for feature in artifact.recalc_features
                if not np.isfinite(pd.to_numeric(row.loc[timestamp, feature], errors="coerce"))
            ]
            if invalid:
                raise ValueError(
                    f"{pair.model_id}/{target}: recursive features are unavailable at "
                    f"{timestamp}: {invalid}"
                )
            x = _transform_for_prediction(estimators[target], row)
            current[target] = float(estimators[target].estimator.predict(x)[0])
            if not np.isfinite(current[target]):
                raise ValueError(
                    f"{pair.model_id}/{target}: estimator returned a non-finite forecast at "
                    f"{timestamp}"
                )

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
        if test_index[0] < start_ts or test_index[-1] > end_ts:
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


def _result_workbook_name(
    family: str,
    model_name: str,
    sheet: int,
    model_id: str,
) -> str:
    if str(family).lower() == "baseline":
        return f"{model_id}.xlsx"
    multi_sheet = {"mutual_lags", "own_lags", "shared_mutual_lags"}
    stem = f"{family}_{model_name}"
    if str(family).lower() in multi_sheet or int(sheet) != 0:
        stem += f"_{int(sheet)}"
    return f"{stem}.xlsx"


def _publication_identity(
    family: str,
    sheet: int,
    lag_policy: str,
    model_id: str,
) -> Tuple[str, str, str]:
    normalized = str(family).lower()
    sheet = int(sheet)
    lag_names = {
        0: ("L1", "lag 1 h"),
        1: ("L24", "lags 1 and 24 h"),
        2: ("L168", "lags 1, 24, and 168 h"),
    }
    lag_code, lag_text = lag_names.get(sheet, (f"L{sheet}", f"preparation sheet {sheet}"))

    if normalized in {"mutual_lags", "shared_mutual_lags"}:
        short = f"MUT-{lag_code}"
        return short, f"Mutual history, {lag_text} ({short})", "Mutual history"
    if normalized == "own_lags":
        short = f"OWN-{lag_code}"
        return short, f"Own history, {lag_text} ({short})", "Own history"
    if normalized == "lagdrop":
        return (
            "No-history",
            "No target history, 38 exogenous predictors (No-history)",
            "No target history",
        )
    if normalized == "mi_top_k_nolags":
        return (
            "MI-9",
            "Mutual-information selection, 9 exogenous predictors (MI-9)",
            "MI-9",
        )
    if normalized == "sfs_shared":
        return "SFS-shared", "SFS, shared P-Q tuning (SFS-shared)", "SFS"
    if normalized == "sfs":
        short = f"SFS-{lag_code}-prep"
        preparation_text = {
            0: "1-h",
            1: "1- and 24-h",
            2: "1-, 24-, and 168-h",
        }.get(sheet, f"sheet-{sheet}")
        return (
            short,
            f"SFS, {preparation_text} preparation matrix ({short})",
            "SFS",
        )
    if normalized == "baseline":
        match = re.search(r"(\d+)", str(lag_policy))
        lag = int(match.group(1)) if match else sheet
        if lag == 24:
            return (
                "SNaive-24",
                "Previous-day seasonal-naive benchmark (SNaive-24)",
                "Seasonal naive",
            )
        if lag == 168:
            return (
                "SNaive-168",
                "Previous-week seasonal-naive benchmark (SNaive-168)",
                "Seasonal naive",
            )

    short = _clean_identifier(model_id)
    return short, short, str(family).replace("_", " ")


def _discover_publication_systems(results_dir: Path) -> Tuple[List[PublicationSystem], pd.DataFrame]:
    summary_path = results_dir / "evaluation_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Publication output requires the aggregate summary: {summary_path}"
        )
    summary = pd.read_csv(summary_path)
    required = {"model_id", "family", "model", "sheet", "lag_policy", "target"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"{summary_path.name} is missing required columns: {missing}")

    systems: List[PublicationSystem] = []
    catalog_rows: List[Dict[str, Any]] = []
    for model_id, group in summary.groupby("model_id", sort=False):
        targets = set(group["target"].astype(str))
        if not set(TARGETS).issubset(targets):
            catalog_rows.append(
                {
                    "model_id": model_id,
                    "status": "excluded",
                    "reason": "incomplete P-Q summary",
                }
            )
            continue
        first = group.iloc[0]
        family = str(first["family"])
        model_name = str(first["model"])
        sheet = int(first["sheet"])
        lag_policy = str(first["lag_policy"])
        workbook = results_dir / _result_workbook_name(
            family, model_name, sheet, str(model_id)
        )
        short, table_label, family_label = _publication_identity(
            family, sheet, lag_policy, str(model_id)
        )
        if not workbook.exists():
            catalog_rows.append(
                {
                    "model_id": model_id,
                    "short_label": short,
                    "family": family,
                    "model": model_name,
                    "sheet": sheet,
                    "lag_policy": lag_policy,
                    "workbook": workbook.name,
                    "table_label": table_label,
                    "family_label": family_label,
                    "status": "excluded",
                    "reason": "result workbook is absent",
                }
            )
            continue
        systems.append(
            PublicationSystem(
                model_id=str(model_id),
                short_label=short,
                table_label=table_label,
                family_label=family_label,
                family=family,
                model_name=model_name,
                sheet=sheet,
                lag_policy=lag_policy,
                workbook=workbook,
            )
        )
        catalog_rows.append(
            {
                "model_id": model_id,
                "short_label": short,
                "table_label": table_label,
                "family_label": family_label,
                "family": family,
                "model": model_name,
                "sheet": sheet,
                "lag_policy": lag_policy,
                "workbook": workbook.name,
                "status": "available",
                "reason": "",
            }
        )
    if not systems:
        raise RuntimeError(f"No complete P-Q result workbooks were found in {results_dir}")
    return systems, pd.DataFrame(catalog_rows)


def _load_result_predictions(
    system: PublicationSystem,
    target: str,
    *,
    expected_horizon: Optional[int] = None,
) -> pd.DataFrame:
    sheet_name = f"{target}_predictions"
    frame = pd.read_excel(
        system.workbook,
        sheet_name=sheet_name,
        usecols=lambda column: column in {"ts", "date", "horizon", "y_true", "y_pred"},
    )
    required = {"ts", "y_true", "y_pred"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{system.workbook.name}/{sheet_name} is missing required columns: {missing}"
        )
    frame["ts"] = pd.to_datetime(frame["ts"], errors="coerce")
    if frame["ts"].isna().any():
        raise ValueError(f"{system.workbook.name}/{sheet_name} contains invalid timestamps")
    frame = frame.sort_values("ts").reset_index(drop=True)
    if frame["ts"].duplicated().any():
        raise ValueError(f"{system.workbook.name}/{sheet_name} contains duplicate timestamps")
    if "date" not in frame.columns:
        frame["date"] = frame["ts"].dt.date
    else:
        dates = pd.to_datetime(frame["date"], errors="coerce")
        if dates.isna().any():
            raise ValueError(f"{system.workbook.name}/{sheet_name} contains invalid dates")
        frame["date"] = dates.dt.date
    if "horizon" not in frame.columns:
        frame["horizon"] = frame.groupby("date", sort=False).cumcount() + 1
    frame["horizon"] = pd.to_numeric(frame["horizon"], errors="coerce")
    for column in ("y_true", "y_pred"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[column].to_numpy(float)).all():
            raise ValueError(
                f"{system.workbook.name}/{sheet_name} contains non-finite {column} values"
            )
    counts = frame.groupby("date", sort=True).size()
    if counts.empty or counts.nunique() != 1:
        raise ValueError(
            f"{system.workbook.name}/{sheet_name} does not contain complete equal-length days"
        )
    horizon = int(counts.iloc[0])
    if expected_horizon is not None and horizon != int(expected_horizon):
        raise ValueError(
            f"{system.workbook.name}/{sheet_name}: expected {expected_horizon} horizons, "
            f"found {horizon}"
        )
    expected = np.arange(1, horizon + 1)
    for date, group in frame.groupby("date", sort=True):
        if not np.array_equal(group["horizon"].to_numpy(int), expected):
            raise ValueError(
                f"{system.workbook.name}/{sheet_name}: invalid horizon sequence on {date}"
            )
    return frame[["ts", "date", "horizon", "y_true", "y_pred"]].copy()


def _prediction_signature(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(frame["ts"].astype("int64").to_numpy(np.int64).tobytes())
    digest.update(frame["y_true"].to_numpy(np.float64).tobytes())
    digest.update(frame["y_pred"].to_numpy(np.float64).tobytes())
    return digest.hexdigest()


def _canonical_system_priority(system: PublicationSystem) -> Tuple[int, str]:
    family = system.family.lower()
    if family == "mutual_lags":
        return 0, system.model_id
    if family == "shared_mutual_lags":
        return 2, system.model_id
    return 1, system.model_id


def _collapse_exact_paired_duplicates(
    systems: Sequence[PublicationSystem],
    *,
    expected_horizon: Optional[int] = None,
) -> Tuple[
    List[PublicationSystem],
    pd.DataFrame,
    Dict[Tuple[str, str], pd.DataFrame],
]:
    predictions: Dict[Tuple[str, str], pd.DataFrame] = {}
    signatures: Dict[Tuple[str, str], str] = {}
    reference: Dict[str, pd.DataFrame] = {}

    for system in systems:
        p_frame = _load_result_predictions(
            system, "P_Power", expected_horizon=expected_horizon
        )
        q_frame = _load_result_predictions(
            system, "Q_Power", expected_horizon=expected_horizon
        )
        if not p_frame["ts"].equals(q_frame["ts"]):
            raise ValueError(f"{system.workbook.name}: P and Q timestamps differ")
        predictions[(system.model_id, "P_Power")] = p_frame
        predictions[(system.model_id, "Q_Power")] = q_frame
        signatures[(system.model_id, "P_Power")] = _prediction_signature(p_frame)
        signatures[(system.model_id, "Q_Power")] = _prediction_signature(q_frame)

        for target, frame in (("P_Power", p_frame), ("Q_Power", q_frame)):
            if target not in reference:
                reference[target] = frame
                continue
            baseline = reference[target]
            if not frame["ts"].equals(baseline["ts"]) or not np.array_equal(
                frame["y_true"].to_numpy(float),
                baseline["y_true"].to_numpy(float),
                equal_nan=True,
            ):
                raise ValueError(
                    f"{system.workbook.name}: evaluation timestamps or actual {target} values "
                    "differ from the other systems"
                )

    groups: Dict[Tuple[str, str], List[PublicationSystem]] = {}
    for system in systems:
        key = (
            signatures[(system.model_id, "P_Power")],
            signatures[(system.model_id, "Q_Power")],
        )
        groups.setdefault(key, []).append(system)

    canonical: List[PublicationSystem] = []
    audit_rows: List[Dict[str, Any]] = []
    for members in groups.values():
        ordered = sorted(members, key=_canonical_system_priority)
        retained = ordered[0]
        canonical.append(retained)
        for member in ordered:
            audit_rows.append(
                {
                    "model_id": member.model_id,
                    "short_label": member.short_label,
                    "workbook": member.workbook.name,
                    "canonical_model_id": retained.model_id,
                    "canonical_short_label": retained.short_label,
                    "exact_paired_duplicate": member.model_id != retained.model_id,
                    "status": (
                        "retained"
                        if member.model_id == retained.model_id
                        else "excluded from complete-system ranking"
                    ),
                }
            )
    canonical.sort(key=lambda item: item.short_label)
    return canonical, pd.DataFrame(audit_rows), predictions


def _complete_system_metrics(
    systems: Sequence[PublicationSystem],
    predictions: Mapping[Tuple[str, str], pd.DataFrame],
    evaluation_summary: pd.DataFrame,
) -> pd.DataFrame:
    summary_lookup = {
        (str(row.model_id), str(row.target)): row
        for row in evaluation_summary.itertuples(index=False)
    }
    rows: List[Dict[str, Any]] = []
    for system in systems:
        row: Dict[str, Any] = {
            "system_id": system.short_label,
            "model_id": system.model_id,
            "table_label": system.table_label,
            "family": system.family_label,
            "workbook": system.workbook.name,
        }
        for short, target in (("P", "P_Power"), ("Q", "Q_Power")):
            frame = predictions[(system.model_id, target)]
            actual = frame["y_true"].to_numpy(float)
            error = actual - frame["y_pred"].to_numpy(float)
            absolute = np.abs(error)
            row[f"{short}_RMSE"] = float(np.sqrt(np.mean(error ** 2)))
            row[f"{short}_MAE"] = float(np.mean(absolute))
            denominator = float(np.sum(np.abs(actual)))
            row[f"{short}_WMAPE%"] = (
                float(100.0 * np.sum(absolute) / denominator)
                if denominator > 1e-12
                else np.nan
            )
            summary_row = summary_lookup.get((system.model_id, target))
            row[f"{short}_MASE"] = (
                float(getattr(summary_row, "MASE"))
                if summary_row is not None and np.isfinite(float(getattr(summary_row, "MASE")))
                else np.nan
            )
        rows.append(row)
    metrics = pd.DataFrame(rows)
    p_min = float(metrics["P_RMSE"].min())
    q_min = float(metrics["Q_RMSE"].min())
    metrics["J"] = 0.5 * (
        metrics["P_RMSE"] / p_min + metrics["Q_RMSE"] / q_min
    )
    return metrics.sort_values(["J", "system_id"]).reset_index(drop=True)


def _configure_publication_plotting(dpi: int) -> None:
    if plt is None:
        raise RuntimeError(
            "Matplotlib is required for publication figures. Install it or use "
            "--no-publication-outputs."
        )
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.25,
            "savefig.dpi": int(dpi),
        }
    )


def _save_publication_figure(
    figure: Any,
    output_dir: Path,
    stem: str,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    paths: List[Path] = []
    for extension in formats:
        normalized = str(extension).lower().lstrip(".")
        if normalized not in {"png", "pdf"}:
            raise ValueError(f"Unsupported figure format: {extension}")
        path = output_dir / f"{stem}.{normalized}"
        figure.savefig(
            path,
            dpi=int(dpi) if normalized == "png" else None,
            bbox_inches="tight",
            pad_inches=0.03,
        )
        paths.append(path)
    return paths


def _plot_all_origin_trajectories(
    selected: PublicationSystem,
    predictions: Mapping[Tuple[str, str], pd.DataFrame],
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
    late_period_days: int,
) -> List[Path]:
    p_frame = predictions[(selected.model_id, "P_Power")]
    q_frame = predictions[(selected.model_id, "Q_Power")]
    fig, axes = plt.subplots(2, 1, figsize=(7.15, 3.35), sharex=True)
    actual_color = "#2b2b2b"
    forecast_color = "#1f5a94"
    last_boundary = p_frame["ts"].max().normalize() + pd.Timedelta(days=1)
    late_start = last_boundary - pd.Timedelta(days=max(0, int(late_period_days)))

    for axis, frame, panel, unit in (
        (axes[0], p_frame, "(a) Active power P", "kW"),
        (axes[1], q_frame, "(b) Reactive power Q", "kVAr"),
    ):
        if late_period_days > 0:
            axis.axvspan(
                late_start, last_boundary, color="#e5e7eb", alpha=0.72, zorder=0
            )
        axis.plot(
            frame["ts"],
            frame["y_true"],
            color=actual_color,
            linewidth=0.75,
            alpha=0.88,
            label="Prepared observation",
            zorder=2,
        )
        axis.plot(
            frame["ts"],
            frame["y_pred"],
            color=forecast_color,
            linewidth=0.82,
            alpha=0.95,
            label=f"{selected.short_label} forecast",
            zorder=3,
        )
        axis.text(
            0.006,
            0.93,
            panel,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            fontweight="bold",
        )
        axis.set_ylabel(unit)
        axis.grid(True, axis="y", color="#d8dce1", linewidth=0.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    if late_period_days > 0:
        label_start = late_start.strftime("%-d %b") if sys.platform != "win32" else late_start.strftime("%#d %b")
        label_end_day = (last_boundary - pd.Timedelta(days=1))
        label_end = (
            label_end_day.strftime("%-d %b")
            if sys.platform != "win32"
            else label_end_day.strftime("%#d %b")
        )
        axes[0].text(
            late_start + (last_boundary - late_start) / 2,
            axes[0].get_ylim()[1],
            f"{label_start}-{label_end}",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color="#4b5563",
        )
    axes[0].legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, frameon=False
    )
    axes[1].set_xlabel("Evaluation timestamp")
    fig.subplots_adjust(left=0.085, right=0.995, top=0.91, bottom=0.14, hspace=0.12)
    stem = f"{selected.short_label.lower().replace('-', '_')}_all_origins_units"
    paths = _save_publication_figure(fig, output_dir, stem, formats, dpi)
    plt.close(fig)
    return paths


def _horizon_rmse(frame: pd.DataFrame) -> pd.Series:
    work = frame.assign(
        squared_error=(frame["y_true"].to_numpy(float) - frame["y_pred"].to_numpy(float)) ** 2
    )
    return np.sqrt(work.groupby("horizon", sort=True)["squared_error"].mean())


def _select_horizon_systems(
    metrics: pd.DataFrame,
    system_by_short: Mapping[str, PublicationSystem],
) -> List[str]:
    preferred = ["MUT-L24", "OWN-L1", "MUT-L168", "No-history", "SNaive-24"]
    selected = [system_id for system_id in preferred if system_id in system_by_short]
    for system_id in metrics["system_id"]:
        if system_id not in selected:
            selected.append(str(system_id))
        if len(selected) == 5:
            break
    return selected


def _plot_horizon_rmse(
    systems: Sequence[PublicationSystem],
    metrics: pd.DataFrame,
    predictions: Mapping[Tuple[str, str], pd.DataFrame],
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    by_short = {system.short_label: system for system in systems}
    plotted = _select_horizon_systems(metrics, by_short)
    style_values = [
        ("#1f5a94", "-", 1.8, "o"),
        ("#2f855a", "-", 1.45, "s"),
        ("#d97706", "--", 1.25, "^"),
        ("#6b7280", "-.", 1.15, "D"),
        ("#111827", ":", 1.25, "x"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.62), sharex=True)
    for axis, target, title, unit in (
        (axes[0], "P_Power", "(a) Active power P", "kW"),
        (axes[1], "Q_Power", "(b) Reactive power Q", "kVAr"),
    ):
        for system_id, style in zip(plotted, style_values):
            color, linestyle, linewidth, marker = style
            system = by_short[system_id]
            series = _horizon_rmse(predictions[(system.model_id, target)])
            axis.plot(
                series.index,
                series.values,
                label=system_id,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                marker=marker,
                markersize=2.5,
                markevery=(0, 3),
            )
        axis.set_title(title, loc="left", fontsize=8.2, fontweight="bold", pad=5)
        axis.set_xlabel("Forecast horizon (h)")
        axis.set_ylabel(f"RMSE ({unit})")
        maximum_horizon = int(max(series.index))
        ticks = sorted({1, 4, 8, 12, 16, 20, maximum_horizon})
        axis.set_xticks([tick for tick in ticks if tick <= maximum_horizon])
        axis.set_xlim(1, maximum_horizon)
        axis.grid(True, color="#d8dce1", linewidth=0.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(5, len(labels)),
        frameon=False,
        handlelength=2.3,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.09, right=0.995, top=0.83, bottom=0.18, wspace=0.24)
    paths = _save_publication_figure(
        fig, output_dir, "horizon_rmse_selected_systems_units", formats, dpi
    )
    plt.close(fig)
    return paths


def _plot_joint_rmse(
    metrics: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    styles = {
        "Mutual history": ("#1f5a94", "s"),
        "Own history": ("#2f855a", "o"),
        "No target history": ("#7b61a8", "D"),
        "SFS": ("#d97706", "^"),
        "MI-9": ("#b94a48", "P"),
        "Seasonal naive": ("#6b7280", "X"),
    }
    fallback = itertools.cycle(
        [("#0f766e", "v"), ("#be123c", "p"), ("#4f46e5", "h")]
    )
    fig, axis = plt.subplots(figsize=(7.15, 3.25))
    for family, group in metrics.groupby("family", sort=False):
        color, marker = styles[family] if family in styles else next(fallback)
        axis.scatter(
            group["P_RMSE"],
            group["Q_RMSE"],
            s=38,
            c=color,
            marker=marker,
            label=family,
            edgecolors="white" if marker not in {"X", "x"} else color,
            linewidths=0.6,
            zorder=3,
        )

    selected = metrics.iloc[0]
    axis.scatter(
        [selected["P_RMSE"]],
        [selected["Q_RMSE"]],
        s=95,
        facecolors="none",
        edgecolors="#111827",
        linewidths=1.0,
        zorder=4,
    )

    for system_id, offset in {
        "No-history": (6, 5),
        "SNaive-24": (6, 5),
        "SNaive-168": (-56, 5),
        "MI-9": (-28, 8),
    }.items():
        match = metrics[metrics["system_id"] == system_id]
        if match.empty:
            continue
        row = match.iloc[0]
        axis.annotate(
            system_id,
            (row["P_RMSE"], row["Q_RMSE"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=6.7,
            color="#1f2937",
            arrowprops=(
                {"arrowstyle": "-", "color": "#9ca3af", "lw": 0.5}
                if abs(offset[0]) > 20
                else None
            ),
        )

    sfs = metrics[metrics["family"] == "SFS"]
    if not sfs.empty:
        axis.annotate(
            f"{len(sfs)} SFS systems",
            (float(sfs["P_RMSE"].mean()), float(sfs["Q_RMSE"].mean())),
            xytext=(-58, 22),
            textcoords="offset points",
            fontsize=6.7,
            color="#1f2937",
            arrowprops={"arrowstyle": "-", "color": "#9ca3af", "lw": 0.5},
        )

    history = metrics[metrics["family"].isin(["Mutual history", "Own history"])]
    if len(history) >= 2:
        inset = axis.inset_axes([0.49, 0.08, 0.31, 0.38])
        for family, group in history.groupby("family", sort=False):
            color, marker = styles[family]
            inset.scatter(
                group["P_RMSE"],
                group["Q_RMSE"],
                s=28,
                c=color,
                marker=marker,
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
            )
        known_offsets = {
            "MUT-L24": (4, 8),
            "MUT-L1": (5, 10),
            "MUT-L168": (-55, -2),
            "OWN-L1": (7, 4),
            "OWN-L24": (-51, -2),
            "OWN-L168": (-49, 6),
        }
        for row in history.itertuples(index=False):
            offset = known_offsets.get(row.system_id, (4, 4))
            inset.annotate(
                row.system_id,
                (row.P_RMSE, row.Q_RMSE),
                xytext=offset,
                textcoords="offset points",
                fontsize=5.8,
                fontweight="bold" if row.system_id == selected["system_id"] else "normal",
                color="#1f2937",
            )
        inset.scatter(
            [selected["P_RMSE"]],
            [selected["Q_RMSE"]],
            s=64,
            facecolors="none",
            edgecolors="#111827",
            linewidths=0.8,
            zorder=4,
        )
        known_history = {
            "MUT-L24", "MUT-L1", "MUT-L168", "OWN-L1", "OWN-L24", "OWN-L168"
        }
        if known_history.issubset(set(history["system_id"])):
            inset.set_xlim(126, 148)
            inset.set_ylim(119, 131)
            inset.set_xticks([128, 136, 144])
            inset.set_yticks([120, 125, 130])
        else:
            p_span = max(
                float(history["P_RMSE"].max() - history["P_RMSE"].min()), 1.0
            )
            q_span = max(
                float(history["Q_RMSE"].max() - history["Q_RMSE"].min()), 1.0
            )
            inset.set_xlim(
                float(history["P_RMSE"].min()) - 0.08 * p_span,
                float(history["P_RMSE"].max()) + 0.18 * p_span,
            )
            inset.set_ylim(
                float(history["Q_RMSE"].min()) - 0.12 * q_span,
                float(history["Q_RMSE"].max()) + 0.18 * q_span,
            )
        inset.tick_params(labelsize=5.7, pad=1)
        inset.grid(True, color="#d8dce1", linewidth=0.4)
        inset.set_title("History-system detail", fontsize=6.2, pad=2)
        axis.indicate_inset_zoom(
            inset, edgecolor="#6b7280", alpha=0.75, linewidth=0.6
        )

    axis.annotate(
        "Lower is better",
        xy=(0.01, 0.03),
        xycoords="axes fraction",
        xytext=(0.16, 0.03),
        textcoords="axes fraction",
        ha="left",
        va="center",
        fontsize=6.8,
        color="#4b5563",
        arrowprops={"arrowstyle": "<|-", "color": "#4b5563", "lw": 0.7},
    )
    axis.set_xlabel("Active-power RMSE (kW)")
    axis.set_ylabel("Reactive-power RMSE (kVAr)")
    axis.grid(True, color="#d8dce1", linewidth=0.5)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    p_span = max(float(metrics["P_RMSE"].max() - metrics["P_RMSE"].min()), 1.0)
    q_span = max(float(metrics["Q_RMSE"].max() - metrics["Q_RMSE"].min()), 1.0)
    axis.set_xlim(
        float(metrics["P_RMSE"].min()) - 0.07 * p_span,
        float(metrics["P_RMSE"].max()) + 0.05 * p_span,
    )
    axis.set_ylim(
        float(metrics["Q_RMSE"].min()) - 0.08 * q_span,
        float(metrics["Q_RMSE"].max()) + 0.08 * q_span,
    )
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=3,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.25,
    )
    fig.subplots_adjust(left=0.09, right=0.995, top=0.98, bottom=0.25)
    paths = _save_publication_figure(
        fig, output_dir, "joint_rmse_complete_systems_units", formats, dpi
    )
    plt.close(fig)
    return paths


def generate_publication_outputs(
    results_dir: str | Path,
    *,
    output_dir: Optional[str | Path] = None,
    figure_formats: Sequence[str] = ("png", "pdf"),
    figure_dpi: int = 600,
    late_period_days: int = 5,
    expected_horizon: Optional[int] = 24,
) -> Dict[str, Any]:
    """Create manuscript tables and figures from completed comparison workbooks."""
    source_dir = Path(results_dir).resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Comparison result directory does not exist: {source_dir}")
    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else source_dir / "publication"
    )
    destination.mkdir(parents=True, exist_ok=True)
    formats = tuple(dict.fromkeys(str(value).lower().lstrip(".") for value in figure_formats))
    if not formats:
        raise ValueError("At least one publication figure format is required")
    if int(figure_dpi) < 72:
        raise ValueError("figure_dpi must be at least 72")
    if int(late_period_days) < 0:
        raise ValueError("late_period_days cannot be negative")

    systems, catalog = _discover_publication_systems(source_dir)
    canonical, duplicate_audit, predictions = _collapse_exact_paired_duplicates(
        systems, expected_horizon=expected_horizon
    )
    evaluation_summary = pd.read_csv(source_dir / "evaluation_summary.csv")
    metrics = _complete_system_metrics(canonical, predictions, evaluation_summary)
    metrics.insert(0, "joint_rank", np.arange(1, len(metrics) + 1, dtype=int))
    metrics["selected_complete_system"] = metrics["joint_rank"].eq(1)
    metrics["minimum_P_RMSE"] = metrics["P_RMSE"].eq(metrics["P_RMSE"].min())
    metrics["minimum_Q_RMSE"] = metrics["Q_RMSE"].eq(metrics["Q_RMSE"].min())
    canonical_ids = set(metrics["model_id"])
    catalog["included_in_complete_system_ranking"] = catalog["model_id"].isin(canonical_ids)
    duplicate_by_id = duplicate_audit.set_index("model_id")["canonical_model_id"].to_dict()
    catalog["canonical_model_id"] = catalog["model_id"].map(duplicate_by_id)

    metrics.to_csv(destination / "complete_system_metrics.csv", index=False)
    manuscript_table = metrics[
        [
            "table_label",
            "P_RMSE",
            "P_MAE",
            "P_WMAPE%",
            "P_MASE",
            "Q_RMSE",
            "Q_MAE",
            "Q_WMAPE%",
            "Q_MASE",
            "J",
        ]
    ].rename(
        columns={
            "table_label": "System",
            "P_RMSE": "P RMSE (kW)",
            "P_MAE": "P MAE (kW)",
            "P_WMAPE%": "P WMAPE (%)",
            "P_MASE": "P MASE",
            "Q_RMSE": "Q RMSE (kVAr)",
            "Q_MAE": "Q MAE (kVAr)",
            "Q_WMAPE%": "Q WMAPE (%)",
            "Q_MASE": "Q MASE",
            "J": "Joint score J",
        }
    )
    manuscript_table = manuscript_table.round(
        {
            "P RMSE (kW)": 2,
            "P MAE (kW)": 2,
            "P WMAPE (%)": 2,
            "P MASE": 3,
            "Q RMSE (kVAr)": 2,
            "Q MAE (kVAr)": 2,
            "Q WMAPE (%)": 2,
            "Q MASE": 3,
            "Joint score J": 4,
        }
    )
    manuscript_table.to_csv(destination / "manuscript_metrics_table.csv", index=False)
    catalog.to_csv(destination / "system_catalog.csv", index=False)
    duplicate_audit.to_csv(destination / "paired_duplicate_audit.csv", index=False)
    with pd.ExcelWriter(destination / "publication_tables.xlsx", engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="complete_system_metrics", index=False)
        manuscript_table.to_excel(writer, sheet_name="manuscript_table", index=False)
        catalog.to_excel(writer, sheet_name="system_catalog", index=False)
        duplicate_audit.to_excel(writer, sheet_name="duplicate_audit", index=False)

    _configure_publication_plotting(int(figure_dpi))
    by_short = {system.short_label: system for system in canonical}
    selected = by_short[str(metrics.iloc[0]["system_id"])]
    generated: List[Path] = []
    generated.extend(
        _plot_all_origin_trajectories(
            selected,
            predictions,
            destination,
            formats,
            int(figure_dpi),
            int(late_period_days),
        )
    )
    generated.extend(
        _plot_horizon_rmse(
            canonical,
            metrics,
            predictions,
            destination,
            formats,
            int(figure_dpi),
        )
    )
    generated.extend(
        _plot_joint_rmse(metrics, destination, formats, int(figure_dpi))
    )

    p_best = metrics.loc[metrics["P_RMSE"].idxmin()]
    q_best = metrics.loc[metrics["Q_RMSE"].idxmin()]
    table_files = [
        destination / "complete_system_metrics.csv",
        destination / "manuscript_metrics_table.csv",
        destination / "system_catalog.csv",
        destination / "paired_duplicate_audit.csv",
        destination / "publication_tables.xlsx",
    ]
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_results_dir": str(source_dir),
        "complete_system_count_before_duplicate_collapse": len(systems),
        "complete_system_count_after_duplicate_collapse": len(canonical),
        "evaluation_days": int(
            predictions[(selected.model_id, "P_Power")]["date"].nunique()
        ),
        "evaluation_rows_per_target": int(
            len(predictions[(selected.model_id, "P_Power")])
        ),
        "forecast_horizon": int(
            predictions[(selected.model_id, "P_Power")]["horizon"].max()
        ),
        "joint_score_definition": (
            "J = 0.5 * (P_RMSE / minimum_P_RMSE + "
            "Q_RMSE / minimum_Q_RMSE)"
        ),
        "selected_complete_system": str(metrics.iloc[0]["system_id"]),
        "selected_joint_score": float(metrics.iloc[0]["J"]),
        "marginal_active_power_minimum": str(p_best["system_id"]),
        "marginal_reactive_power_minimum": str(q_best["system_id"]),
        "late_period_shading_days": int(late_period_days),
        "formal_inference": (
            "Not generated here. Use the separate daily-trajectory "
            "Diebold-Mariano comparison script."
        ),
        "files": {
            path.name: {"sha256": _file_sha256(path), "bytes": path.stat().st_size}
            for path in [*table_files, *generated]
        },
    }
    manifest_path = destination / "publication_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(
        f"Publication outputs: {destination} "
        f"({len(canonical)} unique complete systems; selected "
        f"{metrics.iloc[0]['system_id']})",
        flush=True,
    )
    return manifest


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
    """Write the established four-sheet comparison-workbook contract."""
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


def read_pq_xlsx(path: str) -> List[pd.DataFrame]:
    """Read every PQ workbook sheet and normalize its hourly time index."""
    frames: List[pd.DataFrame] = []
    with pd.ExcelFile(path) as excel:
        for sheet_name in excel.sheet_names:
            frame = excel.parse(sheet_name)
            timestamp_source: Optional[str] = None
            parsed_index: Optional[pd.DatetimeIndex] = None
            if isinstance(frame.index, pd.DatetimeIndex):
                parsed_index = pd.DatetimeIndex(frame.index)
            else:
                candidates: List[Any] = [
                    candidate
                    for candidate in ("Datetime", "datetime", "date", "timestamp")
                    if candidate in frame.columns
                ]
                if len(frame.columns):
                    candidates.append(frame.columns[0])
                for candidate in dict.fromkeys(candidates):
                    parsed = pd.to_datetime(frame[candidate], errors="coerce")
                    if parsed.notna().all():
                        timestamp_source = candidate
                        parsed_index = pd.DatetimeIndex(parsed)
                        break
            if parsed_index is None:
                raise ValueError(f"PQ sheet {sheet_name!r}: no valid timestamp column")
            if parsed_index.isna().any():
                raise ValueError(f"PQ sheet {sheet_name!r}: timestamp parsing produced missing values")
            if timestamp_source is not None:
                frame = frame.drop(columns=[timestamp_source])
            frame.index = parsed_index.floor("h")
            frame.index.name = "ts"
            frame.sort_index(inplace=True)
            if frame.index.has_duplicates:
                duplicates = frame.index[frame.index.duplicated()].unique()[:5]
                raise ValueError(
                    f"PQ sheet {sheet_name!r}: duplicate hourly timestamps, e.g. "
                    f"{list(duplicates)}"
                )
            if len(frame.index) > 1:
                gaps = frame.index.to_series().diff().iloc[1:]
                bad = gaps[gaps != pd.Timedelta(hours=1)]
                if not bad.empty:
                    first = bad.index[0]
                    raise ValueError(
                        f"PQ sheet {sheet_name!r}: non-hourly step before {first} "
                        f"({bad.iloc[0]})"
                    )
            missing_targets = [target for target in TARGETS if target not in frame.columns]
            if missing_targets:
                raise ValueError(
                    f"PQ sheet {sheet_name!r}: missing target columns {missing_targets}"
                )
            for target in TARGETS:
                values = pd.to_numeric(frame[target], errors="coerce").to_numpy(float)
                if not np.isfinite(values).all():
                    raise ValueError(
                        f"PQ sheet {sheet_name!r}: {target} contains missing or non-finite values"
                    )
            frames.append(frame)
    return frames


def _read_pq(path: Path) -> Tuple[List[pd.DataFrame], List[str]]:
    with pd.ExcelFile(path) as excel:
        sheet_names = list(excel.sheet_names)
    sheets = read_pq_xlsx(str(path))
    if len(sheets) != len(sheet_names):
        raise RuntimeError("PQ sheet-name and loaded-sheet counts differ")
    return sheets, sheet_names


def audit_configuration(config: ComparisonConfig) -> Tuple[List[ModelPair], List[str], List[str]]:
    if config.n_jobs < 1:
        raise ValueError("n_jobs must be at least 1")
    pairs, skipped = discover_model_pairs(
        config.models_dir,
        model_types=config.model_types,
        families=config.families,
        sheets=config.sheets,
        include_unprefixed=config.include_unprefixed,
        include_legacy_filenames=config.include_legacy_filenames,
    )
    pairs, duplicate_notes = _collapse_duplicate_model_pairs(pairs)
    skipped.extend(duplicate_notes)
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
            if _pkg.get(package_keys[pair.model_name.upper()]) is None
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

        if config.include_hourly_dm_diagnostics:
            for target in TARGETS:
                short = TARGET_SHORT[target]
                for loss in ("squared", "absolute", "crps", "pinball"):
                    matrices = _pairwise_dm_matrices(
                        predictions_for_dm[target], loss=loss, horizon=config.horizon
                    )
                    loss_code = {
                        "squared": "SE",
                        "absolute": "AE",
                        "crps": "CRPS",
                        "pinball": "PB",
                    }[loss]
                    matrices["stat"].to_excel(
                        writer, sheet_name=f"{short}_{loss_code}_DM_stat"
                    )
                    matrices["p"].to_excel(
                        writer, sheet_name=f"{short}_{loss_code}_DM_p"
                    )
                    matrices["p_holm"].to_excel(
                        writer, sheet_name=f"{short}_{loss_code}_DM_holm"
                    )
                    matrices["n"].to_excel(
                        writer, sheet_name=f"{short}_{loss_code}_DM_n"
                    )

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
                    "complete-system selection",
                    "formal statistical comparison",
                    "optional hourly DM diagnostics",
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
                    "minimum mean target-normalized RMSE over complete paired P-Q systems",
                    "separate daily-trajectory Diebold-Mariano script with multiplicity adjustment",
                    (
                        "included; negative row-vs-column statistic means row model has lower loss"
                        if config.include_hourly_dm_diagnostics
                        else "not included"
                    ),
                ],
            }
        )
        protocol.to_excel(writer, sheet_name="protocol", index=False)

    publication_manifest: Optional[Dict[str, Any]] = None
    if config.publication_outputs:
        publication_manifest = generate_publication_outputs(
            run_dir,
            output_dir=config.publication_dir,
            figure_formats=config.figure_formats,
            figure_dpi=config.figure_dpi,
            late_period_days=config.late_period_days,
            expected_horizon=config.horizon,
        )

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
        "publication_outputs": publication_manifest,
        "methodological_notes": [
            "Prepared PQ.xlsx is read as supplied; preprocessing is not repeated.",
            "Native full-data model files are not used for historical scoring.",
            "October-December 2021 residuals are out-of-sample at each forecast origin but use metadata selected from 2021; they calibrate intervals and are not an independent point-model test.",
            "January-February 2022 is the default evaluation period; later 2022 data are excluded because of the wartime structural break.",
            "Calibration intervals use signed, horizon-specific finite-sample order statistics.",
            "CRPS and quantiles use the exact empirical trajectory ensemble unless bootstrap_simulations > 0.",
            "Whole P/Q residual trajectories preserve temporal and cross-target dependence.",
            "MASE uses one seasonal-naive scale from data preceding the evaluated period.",
            "Complete paired systems are ranked by mean target-normalized RMSE; exact paired forecast duplicates are counted once.",
            "Formal inference is performed separately on daily trajectory losses with multiplicity adjustment.",
            (
                "Optional hourly DM diagnostics use horizon 24, Bartlett HAC, HLN correction, and Holm-adjusted p-values."
                if config.include_hourly_dm_diagnostics
                else "Legacy hourly DM diagnostics were not requested."
            ),
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"\nComparison completed: {run_dir}", flush=True)
    return run_dir


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", default=str(PROJECT_ROOT / "Input" / "PQ.xlsx"))
    parser.add_argument("--models-dir", default=str(PROJECT_ROOT / "Models"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "Proper Model Comparison"))
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
    parser.add_argument(
        "--xgb-one-hot",
        action="store_true",
        help="one-hot encode categorical predictors instead of using native XGBoost categories",
    )
    parser.add_argument("--adaptive-calibration", action="store_true")
    parser.add_argument("--stratify-weekend", action="store_true")
    parser.add_argument("--minimum-pool-days", type=int, default=12)
    parser.add_argument("--bootstrap-simulations", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--include-unprefixed", action="store_true")
    parser.add_argument(
        "--include-legacy-filenames",
        action="store_true",
        help="include metadata using the superseded *_MODEL_<sheet>_* filename convention",
    )
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument(
        "--reject-future-exogenous",
        action="store_true",
        help="exclude models using weather values unless forecast-origin vintages are supplied",
    )
    parser.add_argument("--max-models", type=int)
    parser.add_argument(
        "--figures-from",
        help="regenerate publication outputs from an existing completed result directory",
    )
    parser.add_argument(
        "--publication-dir",
        help="publication output directory; defaults to <run>/publication",
    )
    parser.add_argument(
        "--no-publication-outputs",
        action="store_true",
        help="skip manuscript tables and figures after evaluation",
    )
    parser.add_argument(
        "--figure-formats",
        nargs="+",
        choices=("png", "pdf"),
        default=["png", "pdf"],
    )
    parser.add_argument("--figure-dpi", type=int, default=600)
    parser.add_argument("--late-period-days", type=int, default=5)
    parser.add_argument(
        "--include-hourly-dm-diagnostics",
        action="store_true",
        help="also write the legacy hourly-loss DM matrices to model_comparison.xlsx",
    )
    parser.add_argument("--audit-only", action="store_true")
    return parser


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return _build_cli_parser().parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    _configure_console_errors()
    parser = _build_cli_parser()
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    if not effective_argv:
        parser.print_help()
        return 0
    args = parser.parse_args(effective_argv)
    if args.figures_from:
        if args.audit_only:
            parser.error("--figures-from and --audit-only cannot be used together")
        generate_publication_outputs(
            args.figures_from,
            output_dir=args.publication_dir,
            figure_formats=tuple(args.figure_formats),
            figure_dpi=args.figure_dpi,
            late_period_days=args.late_period_days,
            expected_horizon=args.horizon,
        )
        return 0

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
        xgb_native_categorical=not args.xgb_one_hot,
        adaptive_calibration=args.adaptive_calibration,
        stratify_weekend=args.stratify_weekend,
        minimum_pool_days=args.minimum_pool_days,
        bootstrap_simulations=args.bootstrap_simulations,
        random_seed=args.random_seed,
        include_unprefixed=args.include_unprefixed,
        include_legacy_filenames=args.include_legacy_filenames,
        include_baselines=not args.no_baselines,
        assume_future_exogenous_available=not args.reject_future_exogenous,
        max_models=args.max_models,
        publication_outputs=not args.no_publication_outputs,
        publication_dir=args.publication_dir,
        figure_formats=tuple(args.figure_formats),
        figure_dpi=args.figure_dpi,
        late_period_days=args.late_period_days,
        include_hourly_dm_diagnostics=args.include_hourly_dm_diagnostics,
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

# -*- coding: utf-8 -*-
"""Point-only evaluation for the LightGBM search-mechanism experiment.

Every model specification is refitted at each midnight origin.  Estimator
fitting uses rows no later than the declared fitting cutoff, while target
history remains observed through the preceding hour.  P and Q are then
forecast synchronously for 24 recursive steps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

import Forecasting_search as core
import model_registry


SCRIPT_VERSION = "search-experiment-1.1"
TARGETS = ("P_Power", "Q_Power")
PERIODS = {
    "q4_2021": {
        "start": "2021-10-01 00:00",
        "end": "2021-12-31 00:00",
        "role": (
            "Out-of-search rolling-origin interval for newly searched "
            "configurations; descriptive for HIST because its archived tuning "
            "schedule included later 2021 folds."
        ),
    },
    "prewar_2022": {
        "start": "2022-01-01 00:00",
        "end": "2022-02-23 00:00",
        "role": (
            "Primary rolling-origin comparison conditional on the prepared "
            "target series and supplied future exogenous inputs: 54 midnight origins."
        ),
    },
    "late_february_2022": {
        "start": "2022-02-24 00:00",
        "end": "2022-02-28 00:00",
        "role": "Five-origin structural-stress description; not used for selection.",
    },
}
def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return value


def discover_model_pairs(
    models_dir: Path, requested: Optional[Sequence[str]] = None
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    requested_set = {str(value) for value in requested or ()}
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for path, meta in model_registry.iter_metadata(models_dir):
        snapshot = str(meta.get("experiment_snapshot", ""))
        target = str(meta.get("target", ""))
        if not snapshot.startswith("sx_") or target not in TARGETS:
            continue
        if snapshot.startswith("smoke_"):
            continue
        if requested_set and snapshot not in requested_set:
            continue
        if str(meta.get("model", "")).upper() != "LGBM":
            continue
        if meta.get("lag_policy") != "mutual":
            raise ValueError(f"{path.name}: experiment model must use mutual history")
        if meta.get("feature_selector") != "all":
            raise ValueError(f"{path.name}: experiment model must retain all features")
        missing = [
            feature
            for feature in core.EXPERIMENT_DYNAMIC_FEATURES
            if feature not in meta.get("features", [])
        ]
        if missing:
            raise ValueError(f"{path.name}: missing fixed dynamic features {missing}")
        grouped.setdefault(snapshot, {})[target] = {
            "path": path,
            "sha256": _sha256(path),
            "meta": meta,
        }
    incomplete = {
        snapshot: sorted(set(TARGETS) - set(targets))
        for snapshot, targets in grouped.items()
        if set(targets) != set(TARGETS)
    }
    if incomplete:
        raise ValueError(f"Incomplete experiment model pairs: {incomplete}")
    if requested_set:
        absent = sorted(requested_set - set(grouped))
        if absent:
            raise FileNotFoundError(f"Requested model snapshots are absent: {absent}")
    if not grouped:
        raise RuntimeError(f"No complete experiment model pairs found in {models_dir}")
    return grouped


def _period_origins(
    package_root: Path, period: str, max_origins: Optional[int]
) -> list[dict[str, Any]]:
    if period in PERIODS:
        spec = PERIODS[period]
        days = pd.date_range(spec["start"], spec["end"], freq="D")
        rows = [
            {
                "origin": day,
                "fit_gap_hours": 0,
                "cluster_id": period,
                "event_label": spec["role"],
            }
            for day in days
        ]
    elif period in {"event16", "event16_gaps"}:
        splits = core._read_splits_sheet_xlsx(
            package_root / "Input" / "splits_event16.xlsx", core.EXPERIMENT_SHEET
        )
        rows = []
        gaps = (0,) if period == "event16" else (0, 24, 72, 168)
        for split in splits:
            for gap in gaps:
                rows.append(
                    {
                        "origin": pd.Timestamp(split["test_start"]),
                        "fit_gap_hours": int(gap),
                        "cluster_id": str(split.get("cluster_id", "event")),
                        "event_label": str(split.get("event_label", "")),
                    }
                )
    else:
        raise ValueError(f"Unknown evaluation period: {period}")
    if max_origins is not None:
        rows = rows[: int(max_origins)]
    return rows


def _sanitize_lgbm_params(
    meta: Mapping[str, Any], *, use_gpu: bool, threads: int
) -> Dict[str, Any]:
    tuned = meta.get("tuned_params")
    archived = meta.get("best_params") or {}
    if tuned:
        params = dict(tuned)
        params["subsample_freq"] = int(archived.get("subsample_freq", 1))
        params["random_state"] = int(archived.get("random_state", 42))
    else:
        params = dict(archived)
    for key in (
        "boosting_type",
        "class_weight",
        "importance_type",
        "min_child_weight",
        "min_split_gain",
        "objective",
        "subsample_for_bin",
        "verbose",
        "device",
        "device_type",
        "n_jobs",
    ):
        params.pop(key, None)
    params["n_jobs"] = int(threads)
    params["random_state"] = int(params.get("random_state", 42))
    if not use_gpu:
        params.pop("gpu_platform_id", None)
        params.pop("gpu_device_id", None)
    return params


def _fit_pair(
    *,
    frame: pd.DataFrame,
    X_all: pd.DataFrame,
    y_all: Mapping[str, np.ndarray],
    pair: Mapping[str, Mapping[str, Any]],
    train_start: pd.Timestamp,
    fit_end: pd.Timestamp,
    use_gpu: bool,
    threads: int,
) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    train_mask = (frame.index >= train_start) & (frame.index <= fit_end)
    if train_mask.sum() < 168:
        raise ValueError(f"Only {train_mask.sum()} fitting rows through {fit_end}")
    estimators: Dict[str, Any] = {}
    preprocessing: Dict[str, Dict[str, Any]] = {}
    for target in TARGETS:
        meta = pair[target]["meta"]
        features = list(meta["features"])
        missing = [feature for feature in features if feature not in X_all.columns]
        if missing:
            raise ValueError(f"{target}: features absent from PQ worksheet: {missing}")
        target_mask = train_mask & np.isfinite(y_all[target])
        target_idx = np.flatnonzero(target_mask)
        _, categorical = core.classify_features(
            X_all.iloc[target_idx][features],
            [],
            known_categorical=core.DEFAULT_KNOWN_CATEGORICAL,
        )
        Xfit, preprocessor = core.fit_preprocessor(
            "LGBM",
            X_all.iloc[target_idx][features],
            [feature for feature in features if feature in categorical],
            xgb_native_categorical=True,
        )
        estimator = core.build_estimator(
            "LGBM",
            _sanitize_lgbm_params(meta, use_gpu=use_gpu, threads=threads),
            use_gpu=use_gpu,
        )
        core._fit_fold_estimator(
            estimator,
            model_name="LGBM",
            Xtr=Xfit,
            ytr=y_all[target][target_idx],
            preprocessor=preprocessor,
        )
        estimators[target] = estimator
        preprocessing[target] = preprocessor
    return estimators, preprocessing


def _forecast_origin(
    *,
    frame: pd.DataFrame,
    X_all: pd.DataFrame,
    y_map: Mapping[str, pd.Series],
    pair: Mapping[str, Mapping[str, Any]],
    origin: pd.Timestamp,
    fit_gap_hours: int,
    use_gpu: bool,
    threads: int,
) -> pd.DataFrame:
    history_end = origin - pd.Timedelta(hours=1)
    fit_end = history_end - pd.Timedelta(hours=int(fit_gap_hours))
    train_start = pd.Timestamp("2021-01-02 00:00")
    test_end = origin + pd.Timedelta(hours=23)
    expected = pd.date_range(origin, test_end, freq="h")
    if not expected.isin(frame.index).all():
        raise ValueError(f"Origin {origin} lacks a complete 24-hour test interval")
    if history_end not in frame.index or fit_end not in frame.index:
        raise ValueError(f"Origin {origin} has an unavailable history or fitting cutoff")

    y_all = {
        target: pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
        for target in TARGETS
    }
    estimators, preprocessing = _fit_pair(
        frame=frame,
        X_all=X_all,
        y_all=y_all,
        pair=pair,
        train_start=train_start,
        fit_end=fit_end,
        use_gpu=use_gpu,
        threads=threads,
    )
    allowed_map = {
        target: list(pair[target]["meta"]["features"]) for target in TARGETS
    }
    recalc_map = {
        target: list(pair[target]["meta"]["recalc_features"]) for target in TARGETS
    }
    lag_meta = core.infer_lag_meta(list(X_all.columns), TARGETS)
    core._validate_dynamic_history(
        frame,
        sorted(set(recalc_map["P_Power"] + recalc_map["Q_Power"])),
        lag_meta,
        history_end=history_end,
    )
    recursive = core.roll_predict_multi(
        estimators,
        df_features=X_all,
        targets=TARGETS,
        allowed_map=allowed_map,
        recalc_map=recalc_map,
        lag_meta=lag_meta,
        history_end=history_end,
        test_start=origin,
        test_end=test_end,
        y_map=dict(y_map),
        preprocessor_map=preprocessing,
    )
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        for horizon, timestamp in enumerate(expected, start=1):
            prediction = float(recursive[target][timestamp])
            actual = float(frame.at[timestamp, target])
            rows.append(
                {
                    "origin": origin,
                    "timestamp": timestamp,
                    "horizon": horizon,
                    "target": target,
                    "actual": actual,
                    "prediction": prediction,
                    "error": prediction - actual,
                    "fit_gap_hours": int(fit_gap_hours),
                    "fit_end": fit_end,
                    "history_end": history_end,
                }
            )
    return pd.DataFrame(rows)


def _seasonal_naive_predictions(
    frame: pd.DataFrame,
    origins: Sequence[Mapping[str, Any]],
    *,
    lag: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in origins:
        origin = pd.Timestamp(item["origin"])
        for horizon, timestamp in enumerate(
            pd.date_range(origin, periods=24, freq="h"), start=1
        ):
            source = timestamp - pd.Timedelta(hours=int(lag))
            for target in TARGETS:
                rows.append(
                    {
                        "origin": origin,
                        "timestamp": timestamp,
                        "horizon": horizon,
                        "target": target,
                        "actual": float(frame.at[timestamp, target]),
                        "prediction": float(frame.at[source, target]),
                        "fit_gap_hours": int(item["fit_gap_hours"]),
                        "fit_end": pd.NaT,
                        "history_end": origin - pd.Timedelta(hours=1),
                        "cluster_id": str(item["cluster_id"]),
                        "event_label": str(item["event_label"]),
                    }
                )
    result = pd.DataFrame(rows)
    result["error"] = result["prediction"] - result["actual"]
    return result


def _mase_scale(series: pd.Series, start: pd.Timestamp, lag: int = 24) -> float:
    values = pd.to_numeric(series.loc[series.index < start], errors="coerce").dropna()
    if len(values) <= lag:
        return math.nan
    difference = np.abs(values.to_numpy()[lag:] - values.to_numpy()[:-lag])
    return float(np.mean(difference))


def _summarize_predictions(
    *,
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    snapshot: str,
    period: str,
    scales: Mapping[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    period_start = pd.Timestamp(predictions["timestamp"].min())
    for (gap, target), group in predictions.groupby(
        ["fit_gap_hours", "target"], sort=True
    ):
        error = group["error"].to_numpy(dtype=float)
        actual = group["actual"].to_numpy(dtype=float)
        rmse = float(np.sqrt(np.mean(error**2)))
        mae = float(np.mean(np.abs(error)))
        wmape = float(100.0 * np.sum(np.abs(error)) / np.sum(np.abs(actual)))
        mase_denominator = _mase_scale(frame[target], period_start, lag=24)
        summary_rows.append(
            {
                "snapshot": snapshot,
                "period": period,
                "fit_gap_hours": int(gap),
                "target": target,
                "n_origins": int(group["origin"].nunique()),
                "n_hours": int(len(group)),
                "RMSE": rmse,
                "MAE": mae,
                "WMAPE_percent": wmape,
                "MASE": (
                    float(mae / mase_denominator)
                    if np.isfinite(mase_denominator) and mase_denominator > 0
                    else math.nan
                ),
            }
        )
    for (gap, origin), day in predictions.groupby(
        ["fit_gap_hours", "origin"], sort=True
    ):
        cluster_ids = day["cluster_id"].dropna().astype(str).unique()
        event_labels = day["event_label"].dropna().astype(str).unique()
        if len(cluster_ids) != 1 or len(event_labels) != 1:
            raise ValueError(
                f"Inconsistent event metadata at {origin}, gap {gap}: "
                f"clusters={cluster_ids}, labels={event_labels}"
            )
        by_target = {target: day[day["target"] == target] for target in TARGETS}
        if any(len(by_target[target]) != 24 for target in TARGETS):
            raise ValueError(f"Incomplete paired trajectory at {origin}, gap {gap}")
        p_error = by_target["P_Power"]["error"].to_numpy(dtype=float)
        q_error = by_target["Q_Power"]["error"].to_numpy(dtype=float)
        p_mse = float(np.mean(p_error**2))
        q_mse = float(np.mean(q_error**2))
        daily_rows.append(
            {
                "snapshot": snapshot,
                "period": period,
                "fit_gap_hours": int(gap),
                "origin": origin,
                "cluster_id": cluster_ids[0],
                "event_label": event_labels[0],
                "P_MSE": p_mse,
                "Q_MSE": q_mse,
                "paired_normalized_squared_loss": 0.5
                * (
                    p_mse / float(scales["P_Power"]) ** 2
                    + q_mse / float(scales["Q_Power"]) ** 2
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    joint_rows = []
    for gap, group in summary.groupby("fit_gap_hours", sort=True):
        p = group[group["target"] == "P_Power"].iloc[0]
        q = group[group["target"] == "Q_Power"].iloc[0]
        joint_rows.append(
            {
                "snapshot": snapshot,
                "period": period,
                "fit_gap_hours": int(gap),
                "target": "PAIRED",
                "n_origins": int(p["n_origins"]),
                "n_hours": int(p["n_hours"] + q["n_hours"]),
                "RMSE": math.nan,
                "MAE": math.nan,
                "WMAPE_percent": math.nan,
                "MASE": math.nan,
                "joint_normalized_RMSE": 0.5
                * (
                    float(p["RMSE"]) / float(scales["P_Power"])
                    + float(q["RMSE"]) / float(scales["Q_Power"])
                ),
            }
        )
    summary["joint_normalized_RMSE"] = math.nan
    summary = pd.concat([summary, pd.DataFrame(joint_rows)], ignore_index=True)
    return summary, pd.DataFrame(daily_rows)


def _event_summary_tables(
    daily: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event = daily[
        (daily["period"] == "event16") & (daily["fit_gap_hours"] == 0)
    ].copy()
    if event.empty:
        cluster_metrics = pd.DataFrame()
    else:
        cluster_metrics = (
            event.groupby(
                ["snapshot", "cluster_id", "event_label"], as_index=False
            )
            .agg(
                n_origins=("origin", "nunique"),
                mean_daily_loss=("paired_normalized_squared_loss", "mean"),
                median_daily_loss=("paired_normalized_squared_loss", "median"),
                maximum_daily_loss=("paired_normalized_squared_loss", "max"),
            )
            .sort_values(["snapshot", "mean_daily_loss"], ascending=[True, False])
        )
        cluster_metrics["difficulty_rank_within_snapshot"] = (
            cluster_metrics.groupby("snapshot")["mean_daily_loss"]
            .rank(method="min", ascending=False)
            .astype(int)
        )

    gaps = daily[daily["period"] == "event16_gaps"].copy()
    if gaps.empty:
        gap_degradation = pd.DataFrame()
    else:
        gap_degradation = (
            gaps.groupby(
                [
                    "snapshot",
                    "cluster_id",
                    "event_label",
                    "fit_gap_hours",
                ],
                as_index=False,
            )
            .agg(
                n_origins=("origin", "nunique"),
                mean_daily_loss=("paired_normalized_squared_loss", "mean"),
                maximum_daily_loss=("paired_normalized_squared_loss", "max"),
            )
            .sort_values(["snapshot", "cluster_id", "fit_gap_hours"])
        )
        baseline = (
            gap_degradation[gap_degradation["fit_gap_hours"] == 0][
                ["snapshot", "cluster_id", "mean_daily_loss"]
            ]
            .rename(columns={"mean_daily_loss": "gap_0_mean_daily_loss"})
            .copy()
        )
        gap_degradation = gap_degradation.merge(
            baseline, on=["snapshot", "cluster_id"], how="left", validate="many_to_one"
        )
        gap_degradation["absolute_degradation_from_gap_0"] = (
            gap_degradation["mean_daily_loss"]
            - gap_degradation["gap_0_mean_daily_loss"]
        )
        denominator = gap_degradation["gap_0_mean_daily_loss"].replace(0.0, np.nan)
        gap_degradation["percent_degradation_from_gap_0"] = (
            100.0
            * gap_degradation["absolute_degradation_from_gap_0"]
            / denominator
        )
    return cluster_metrics, gap_degradation


def _checkpoint_fingerprint(
    *,
    pq_path: Path,
    pair: Mapping[str, Mapping[str, Any]],
    period: str,
    origins: Sequence[Mapping[str, Any]],
    use_gpu: bool,
    threads: int,
) -> str:
    payload = {
        "script_version": SCRIPT_VERSION,
        "evaluation_script_sha256": _sha256(Path(__file__).resolve()),
        "forecasting_script_sha256": _sha256(
            Path(core.__file__).resolve()
        ),
        "pq_sha256": _sha256(pq_path),
        "objective_scales_sha256": _sha256(
            pq_path.parent / "objective_scales.json"
        ),
        "model_metadata_sha256": {
            target: pair[target]["sha256"] for target in TARGETS
        },
        "period": period,
        "origins": [
            {
                "origin": pd.Timestamp(item["origin"]),
                "fit_gap_hours": int(item["fit_gap_hours"]),
                "cluster_id": str(item["cluster_id"]),
                "event_label": str(item["event_label"]),
            }
            for item in origins
        ],
        "device": "gpu" if use_gpu else "cpu",
        "threads": int(threads),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        ).encode("utf-8")
    ).hexdigest()


def evaluate_snapshot_period(
    *,
    package_root: Path,
    frame: pd.DataFrame,
    pair: Mapping[str, Mapping[str, Any]],
    snapshot: str,
    period: str,
    checkpoint_namespace: str,
    origins: Sequence[Mapping[str, Any]],
    use_gpu: bool,
    threads: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir = (
        package_root
        / "_work"
        / "evaluation"
        / period
        / checkpoint_namespace
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{snapshot}_predictions.csv"
    manifest_path = output_dir / f"{snapshot}_manifest.json"
    fingerprint = _checkpoint_fingerprint(
        pq_path=package_root / "Input" / "PQ.xlsx",
        pair=pair,
        period=period,
        origins=origins,
        use_gpu=use_gpu,
        threads=threads,
    )
    manifest = _read_json(manifest_path) if manifest_path.exists() else None
    if manifest is not None and manifest.get("fingerprint") != fingerprint:
        raise RuntimeError(
            f"Evaluation checkpoint changed for {snapshot}/{period}: {manifest_path}"
        )
    predictions = (
        pd.read_csv(
            checkpoint_path,
            parse_dates=["origin", "timestamp", "fit_end", "history_end"],
        )
        if checkpoint_path.exists()
        else pd.DataFrame()
    )
    if manifest is None:
        _write_json(
            manifest_path,
            {
                "fingerprint": fingerprint,
                "script_version": SCRIPT_VERSION,
                "snapshot": snapshot,
                "period": period,
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    completed = (
        {
            (pd.Timestamp(origin), int(gap))
            for origin, gap in predictions[["origin", "fit_gap_hours"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        }
        if not predictions.empty
        else set()
    )
    X_all = frame.drop(columns=list(TARGETS))
    y_map = {target: frame[target] for target in TARGETS}
    for number, item in enumerate(origins, start=1):
        origin = pd.Timestamp(item["origin"])
        gap = int(item["fit_gap_hours"])
        if (origin, gap) in completed:
            continue
        forecast = _forecast_origin(
            frame=frame,
            X_all=X_all,
            y_map=y_map,
            pair=pair,
            origin=origin,
            fit_gap_hours=gap,
            use_gpu=use_gpu,
            threads=threads,
        )
        forecast["cluster_id"] = str(item["cluster_id"])
        forecast["event_label"] = str(item["event_label"])
        predictions = pd.concat([predictions, forecast], ignore_index=True)
        predictions.sort_values(
            ["fit_gap_hours", "origin", "target", "horizon"], inplace=True
        )
        temporary = checkpoint_path.with_suffix(".tmp.csv")
        predictions.to_csv(temporary, index=False)
        os.replace(temporary, checkpoint_path)
        print(
            f"[{snapshot}/{period}] {number}/{len(origins)} "
            f"origin={origin.date()} fit_gap={gap}h"
        )
    scales_json = _read_json(package_root / "Input" / "objective_scales.json")
    scales = {
        target: float(scales_json["target_scales"][target]) for target in TARGETS
    }
    summary, daily = _summarize_predictions(
        frame=frame,
        predictions=predictions,
        snapshot=snapshot,
        period=period,
        scales=scales,
    )
    _write_json(
        manifest_path,
        {
            "fingerprint": fingerprint,
            "script_version": SCRIPT_VERSION,
            "snapshot": snapshot,
            "period": period,
            "status": "complete",
            "origin_count": len(origins),
            "forecast_rows": len(predictions),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "model_metadata": {
                target: {
                    "path": str(pair[target]["path"].resolve()),
                    "sha256": pair[target]["sha256"],
                }
                for target in TARGETS
            },
        },
    )
    return predictions, summary, daily


def run_evaluation(
    *,
    package_root: Path,
    periods: Sequence[str],
    requested_models: Optional[Sequence[str]],
    include_baselines: bool,
    use_gpu: bool,
    threads: int,
    max_origins: Optional[int],
    dry_run: bool,
    output_namespace: Optional[Path] = None,
    checkpoint_namespace: Optional[str] = None,
) -> Dict[str, Any]:
    pq_path = package_root / "Input" / "PQ.xlsx"
    frame = core._read_pq_sheet_xlsx(pq_path, core.EXPERIMENT_SHEET)
    pairs = discover_model_pairs(package_root / "Models", requested_models)
    all_summary: list[pd.DataFrame] = []
    all_daily: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    plan = {
        "script_version": SCRIPT_VERSION,
        "models": sorted(pairs),
        "periods": list(periods),
        "device": "gpu" if use_gpu else "cpu",
        "threads": int(threads),
        "dry_run": bool(dry_run),
        "output_namespace": (
            output_namespace.as_posix() if output_namespace is not None else None
        ),
        "checkpoint_namespace": checkpoint_namespace or "production",
        "jobs": [],
    }
    for period in periods:
        origins = _period_origins(package_root, period, max_origins)
        for snapshot, pair in sorted(pairs.items()):
            plan["jobs"].append(
                {
                    "snapshot": snapshot,
                    "period": period,
                    "period_role": (
                        PERIODS[period]["role"]
                        if period in PERIODS
                        else (
                            "Curated event subset; dates overlap the standard "
                            "rolling-origin periods and are not an independent holdout."
                        )
                    ),
                    "origin_gap_rows": len(origins),
                    "target_origin_fits": 2 * len(origins),
                }
            )
            if dry_run:
                continue
            predictions, summary, daily = evaluate_snapshot_period(
                package_root=package_root,
                frame=frame,
                pair=pair,
                snapshot=snapshot,
                period=period,
                checkpoint_namespace=(
                    checkpoint_namespace or "production"
                    if max_origins is None
                    else f"smoke_n{int(max_origins)}"
                ),
                origins=origins,
                use_gpu=use_gpu,
                threads=threads,
            )
            all_predictions.append(predictions.assign(snapshot=snapshot, period=period))
            all_summary.append(summary)
            all_daily.append(daily)
        if include_baselines and not dry_run:
            for lag in (24, 168):
                snapshot = f"SNaive-{lag}"
                predictions = _seasonal_naive_predictions(frame, origins, lag=lag)
                predictions["snapshot"] = snapshot
                predictions["period"] = period
                scales_json = _read_json(
                    package_root / "Input" / "objective_scales.json"
                )
                scales = {
                    target: float(scales_json["target_scales"][target])
                    for target in TARGETS
                }
                summary, daily = _summarize_predictions(
                    frame=frame,
                    predictions=predictions,
                    snapshot=snapshot,
                    period=period,
                    scales=scales,
                )
                all_predictions.append(predictions)
                all_summary.append(summary)
                all_daily.append(daily)
    if output_namespace is None:
        manifest_path = package_root / "_work" / "provenance" / "evaluation_plan.json"
    else:
        manifest_path = (
            package_root
            / "_work"
            / "provenance"
            / output_namespace
            / "evaluation_plan.json"
        )
    _write_json(manifest_path, plan)
    if dry_run:
        print(
            f"[evaluation dry run] models={len(pairs)} periods={len(periods)} "
            f"planned_jobs={len(plan['jobs'])}"
        )
        return plan

    output_root = package_root / "_work"
    if output_namespace is None:
        output_root = output_root / "forecasting"
    else:
        output_root = output_root / output_namespace
    if max_origins is not None:
        output_root = output_root / f"smoke_n{int(max_origins)}"
    output_root.mkdir(parents=True, exist_ok=True)
    summary_table = pd.concat(all_summary, ignore_index=True)
    daily_table = pd.concat(all_daily, ignore_index=True)
    prediction_table = pd.concat(all_predictions, ignore_index=True)
    cluster_metrics, gap_degradation = _event_summary_tables(daily_table)
    summary_table.to_csv(output_root / "metrics_summary.csv", index=False)
    daily_table.to_csv(output_root / "daily_paired_losses.csv", index=False)
    prediction_table.to_csv(output_root / "predictions.csv", index=False)
    with pd.ExcelWriter(output_root / "model_comparison.xlsx") as writer:
        summary_table.to_excel(writer, sheet_name="summary", index=False)
        daily_table.to_excel(writer, sheet_name="daily_paired_loss", index=False)
        if not cluster_metrics.empty:
            cluster_metrics.to_excel(
                writer, sheet_name="event16_clusters", index=False
            )
        if not gap_degradation.empty:
            gap_degradation.to_excel(
                writer, sheet_name="event16_gap_effects", index=False
            )
    tables_root = (
        package_root / "_work" / "tables"
        if output_namespace is None
        else output_root / "tables"
    )
    tables_root.mkdir(parents=True, exist_ok=True)
    if not cluster_metrics.empty:
        cluster_metrics.to_csv(
            tables_root / "event16_cluster_metrics.csv", index=False
        )
    if not gap_degradation.empty:
        gap_degradation.to_csv(
            tables_root / "event16_gap_degradation.csv", index=False
        )
    plan.update(
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "output_files": {
                "summary": str((output_root / "metrics_summary.csv").resolve()),
                "daily": str((output_root / "daily_paired_losses.csv").resolve()),
                "predictions": str((output_root / "predictions.csv").resolve()),
                "event16_cluster_metrics": (
                    str(
                        (
                            tables_root / "event16_cluster_metrics.csv"
                        ).resolve()
                    )
                    if not cluster_metrics.empty
                    else None
                ),
                "event16_gap_degradation": (
                    str(
                        (
                            tables_root / "event16_gap_degradation.csv"
                        ).resolve()
                    )
                    if not gap_degradation.empty
                    else None
                ),
                "workbook": str((output_root / "model_comparison.xlsx").resolve()),
            },
        }
    )
    _write_json(manifest_path, plan)
    return plan


def _build_parser() -> argparse.ArgumentParser:
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=package_root)
    parser.add_argument(
        "--periods",
        nargs="+",
        choices=[*PERIODS, "event16", "event16_gaps", "all"],
        default=["prewar_2022"],
    )
    parser.add_argument("--models", nargs="+")
    parser.add_argument(
        "--output-namespace",
        type=Path,
        help=(
            "Relative directory below _work for consolidated outputs. "
            "Its evaluation manifest is stored below _work/provenance using the same "
            "namespace. Omit to retain the standard _work/forecasting layout."
        ),
    )
    parser.add_argument(
        "--checkpoint-namespace",
        help=(
            "Isolated evaluation checkpoint name. Omit to retain the shared "
            "production checkpoint namespace."
        ),
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "gpu"], default="auto"
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-origins", type=int)
    parser.add_argument(
        "--no-baselines", action="store_true", help="Do not add SNaive-24/168."
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = _build_parser().parse_args(argv)
    package_root = args.package_root.expanduser().resolve()
    output_namespace = args.output_namespace
    checkpoint_namespace = args.checkpoint_namespace
    if checkpoint_namespace is not None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", checkpoint_namespace):
            raise ValueError(
                "--checkpoint-namespace may contain only letters, numbers, "
                "periods, underscores, and hyphens"
            )
    if output_namespace is not None:
        if output_namespace.is_absolute() or ".." in output_namespace.parts:
            raise ValueError("--output-namespace must be a relative in-project path")
        output_namespace = Path(
            *[part for part in output_namespace.parts if part not in ("", ".")]
        )
        if not output_namespace.parts:
            raise ValueError("--output-namespace must not be empty")
    periods = list(args.periods)
    if "all" in periods:
        periods = [*PERIODS, "event16", "event16_gaps"]
    if args.dry_run and args.device == "auto":
        selection = package_root / "Input" / "device_selection.json"
        use_gpu = (
            selection.exists()
            and str(_read_json(selection)["selected_device"]).lower() == "gpu"
        )
    else:
        use_gpu = core._resolve_experiment_device(
            package_root=package_root,
            requested=args.device,
            threads=int(args.threads),
        )
    run_evaluation(
        package_root=package_root,
        periods=periods,
        requested_models=args.models,
        include_baselines=not args.no_baselines,
        use_gpu=use_gpu,
        threads=int(args.threads),
        max_origins=args.max_origins,
        dry_run=bool(args.dry_run),
        output_namespace=output_namespace,
        checkpoint_namespace=checkpoint_namespace,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

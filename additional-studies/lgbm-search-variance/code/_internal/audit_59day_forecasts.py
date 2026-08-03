# -*- coding: utf-8 -*-
"""Audit archived 59-origin forecasts and the recursive information boundary.

This script does not tune models or create new candidate specifications.  It
checks archived forecast structure, then refits three frozen T60 specifications
at three selected origins to verify future-target perturbation and target-order
invariance of the recursive forecast implementation.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


TARGETS = ("P_Power", "Q_Power")
CONFIG_SOURCES = {
    "S24-T10": ("_work/forecasting/predictions.csv", "sx_norm_t10"),
    "S24-T30": ("_work/forecasting/predictions.csv", "sx_norm_t30"),
    "S24-T60": ("_work/breadth_t60/point/predictions.csv", "sx_norm_t60"),
    "S36-T60": (
        "_work/breadth_t60/point/predictions.csv",
        "sx_norm_s36_t60",
    ),
    "S48-T60": (
        "_work/breadth_t60/point/predictions.csv",
        "sx_norm_s48_t60",
    ),
}
BEHAVIORAL_CONFIGS = {
    "S24-T60": "sx_norm_t60",
    "S36-T60": "sx_norm_s36_t60",
    "S48-T60": "sx_norm_s48_t60",
}
AUDIT_ORIGINS = tuple(
    pd.Timestamp(value) for value in ("2022-01-01", "2022-02-01", "2022-02-24")
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _load_archived(package_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    cache: dict[Path, pd.DataFrame] = {}
    for config, (relative, snapshot) in CONFIG_SOURCES.items():
        path = package_root / relative
        if path not in cache:
            cache[path] = pd.read_csv(
                path,
                parse_dates=["origin", "timestamp", "fit_end", "history_end"],
            )
        source = cache[path]
        selected = source.loc[
            (source["snapshot"] == snapshot)
            & source["period"].isin(["prewar_2022", "late_february_2022"])
        ].copy()
        selected["configuration"] = config
        frames.append(selected)
    archived = pd.concat(frames, ignore_index=True)
    archived["analysis_period"] = np.where(
        archived["origin"] <= pd.Timestamp("2022-02-23"),
        "primary_54",
        "descriptive_stress_5",
    )
    return archived


def _structural_audit(archived: pd.DataFrame) -> tuple[list[dict[str, Any]], float]:
    details: list[dict[str, Any]] = []
    actual_reference: pd.Series | None = None
    maximum_actual_difference = 0.0

    for config in CONFIG_SOURCES:
        subset = archived.loc[archived["configuration"] == config].copy()
        origins = pd.DatetimeIndex(sorted(subset["origin"].unique()))
        assert len(origins) == 59, (config, len(origins))
        assert origins[0] == pd.Timestamp("2022-01-01")
        assert origins[-1] == pd.Timestamp("2022-02-28")
        assert subset["prediction"].notna().all()
        assert np.isfinite(subset["prediction"].to_numpy(dtype=float)).all()
        assert not subset.duplicated(["target", "timestamp"]).any()
        assert (subset["fit_gap_hours"].astype(int) == 0).all()
        assert (subset["fit_end"] == subset["origin"] - pd.Timedelta(hours=1)).all()
        assert (
            subset["history_end"] == subset["origin"] - pd.Timedelta(hours=1)
        ).all()

        target_keys: dict[str, pd.DataFrame] = {}
        for target in TARGETS:
            target_rows = subset.loc[subset["target"] == target].copy()
            assert len(target_rows) == 1416, (config, target, len(target_rows))
            assert target_rows["origin"].nunique() == 59
            assert set(target_rows["horizon"].astype(int).unique()) == set(
                range(1, 25)
            )
            per_origin = target_rows.groupby("origin").size()
            assert (per_origin == 24).all()
            expected_timestamps = target_rows["origin"] + pd.to_timedelta(
                target_rows["horizon"].astype(int) - 1, unit="h"
            )
            assert (target_rows["timestamp"] == expected_timestamps).all()
            target_keys[target] = target_rows[
                ["origin", "timestamp", "horizon", "fit_end", "history_end"]
            ].sort_values(["origin", "horizon"]).reset_index(drop=True)

        pd.testing.assert_frame_equal(
            target_keys[TARGETS[0]], target_keys[TARGETS[1]], check_dtype=False
        )

        actuals = (
            subset[["target", "timestamp", "actual"]]
            .sort_values(["target", "timestamp"])
            .set_index(["target", "timestamp"])["actual"]
        )
        if actual_reference is None:
            actual_reference = actuals
        else:
            aligned = actual_reference.to_frame("reference").join(
                actuals.rename("candidate"), how="inner"
            )
            difference = float(
                np.max(np.abs(aligned["reference"] - aligned["candidate"]))
            )
            maximum_actual_difference = max(maximum_actual_difference, difference)
            assert difference < 1e-12, (config, difference)

        details.append(
            {
                "configuration": config,
                "origins": 59,
                "hours_per_origin_per_target": 24,
                "rows_per_target": 1416,
                "first_origin": origins[0],
                "last_origin": origins[-1],
                "fit_end_rule_verified": True,
                "history_end_rule_verified": True,
                "future_target_rows_in_fit_set": 0,
                "missing_predictions": 0,
                "nonfinite_predictions": 0,
                "duplicate_target_timestamp_rows": 0,
                "paired_target_timestamps_verified": True,
            }
        )
    return details, maximum_actual_difference


def _predict_with_order(
    *,
    core: Any,
    estimators: Mapping[str, Any],
    preprocessing: Mapping[str, Mapping[str, Any]],
    X_all: pd.DataFrame,
    pair: Mapping[str, Mapping[str, Any]],
    y_map: Mapping[str, pd.Series],
    origin: pd.Timestamp,
    targets: Sequence[str],
) -> dict[str, dict[pd.Timestamp, float]]:
    allowed_map = {
        target: list(pair[target]["meta"]["features"]) for target in TARGETS
    }
    recalc_map = {
        target: list(pair[target]["meta"]["recalc_features"]) for target in TARGETS
    }
    lag_meta = core.infer_lag_meta(list(X_all.columns), TARGETS)
    history_end = origin - pd.Timedelta(hours=1)
    return core.roll_predict_multi(
        dict(estimators),
        df_features=X_all,
        targets=list(targets),
        allowed_map=allowed_map,
        recalc_map=recalc_map,
        lag_meta=lag_meta,
        history_end=history_end,
        test_start=origin,
        test_end=origin + pd.Timedelta(hours=23),
        y_map=dict(y_map),
        preprocessor_map=dict(preprocessing),
    )


def _max_prediction_difference(
    first: Mapping[str, Mapping[pd.Timestamp, float]],
    second: Mapping[str, Mapping[pd.Timestamp, float]],
) -> float:
    differences = [
        abs(float(first[target][timestamp]) - float(second[target][timestamp]))
        for target in TARGETS
        for timestamp in first[target]
    ]
    return float(max(differences, default=0.0))


def _behavioral_audit(
    package_root: Path, *, threads: int
) -> tuple[list[dict[str, Any]], float, float]:
    code_dir = package_root / "code"
    if (code_dir / "_internal").is_dir():
        code_dir = code_dir / "_internal"
    sys.path.insert(0, str(code_dir))
    import Forecasting_search as core  # pylint: disable=import-outside-toplevel
    import evaluate_search as evaluator  # pylint: disable=import-outside-toplevel

    frame = core._read_pq_sheet_xlsx(
        package_root / "Input" / "PQ.xlsx", core.EXPERIMENT_SHEET
    )
    X_all = frame.drop(columns=list(TARGETS))
    original_y = {target: frame[target].copy() for target in TARGETS}
    pairs = evaluator.discover_model_pairs(
        package_root / "Models", list(BEHAVIORAL_CONFIGS.values())
    )
    rows: list[dict[str, Any]] = []
    maximum_perturbation = 0.0
    maximum_order = 0.0

    for config, snapshot in BEHAVIORAL_CONFIGS.items():
        pair = pairs[snapshot]
        for origin in AUDIT_ORIGINS:
            history_end = origin - pd.Timedelta(hours=1)
            y_all = {
                target: pd.to_numeric(frame[target], errors="coerce").to_numpy(
                    dtype=float
                )
                for target in TARGETS
            }
            estimators, preprocessing = evaluator._fit_pair(
                frame=frame,
                X_all=X_all,
                y_all=y_all,
                pair=pair,
                train_start=pd.Timestamp("2021-01-02 00:00"),
                fit_end=history_end,
                use_gpu=False,
                threads=threads,
            )
            baseline = _predict_with_order(
                core=core,
                estimators=estimators,
                preprocessing=preprocessing,
                X_all=X_all,
                pair=pair,
                y_map=original_y,
                origin=origin,
                targets=TARGETS,
            )
            perturbed_y = {target: series.copy() for target, series in original_y.items()}
            for target in TARGETS:
                perturbed_y[target].loc[perturbed_y[target].index >= origin] = 9.87654321e12
            perturbed = _predict_with_order(
                core=core,
                estimators=estimators,
                preprocessing=preprocessing,
                X_all=X_all,
                pair=pair,
                y_map=perturbed_y,
                origin=origin,
                targets=TARGETS,
            )
            reversed_order = _predict_with_order(
                core=core,
                estimators=estimators,
                preprocessing=preprocessing,
                X_all=X_all,
                pair=pair,
                y_map=original_y,
                origin=origin,
                targets=tuple(reversed(TARGETS)),
            )
            perturbation_difference = _max_prediction_difference(baseline, perturbed)
            order_difference = _max_prediction_difference(baseline, reversed_order)
            maximum_perturbation = max(maximum_perturbation, perturbation_difference)
            maximum_order = max(maximum_order, order_difference)
            rows.append(
                {
                    "configuration": config,
                    "snapshot": snapshot,
                    "origin": origin,
                    "fit_end": history_end,
                    "history_end": history_end,
                    "future_target_perturbation_difference": perturbation_difference,
                    "target_order_difference": order_difference,
                    "perturbation_pass": perturbation_difference < 1e-10,
                    "target_order_pass": order_difference < 1e-10,
                }
            )
            assert perturbation_difference < 1e-10, rows[-1]
            assert order_difference < 1e-10, rows[-1]
            print(
                f"[audit] {config} {origin.date()} "
                f"perturb={perturbation_difference:.3g} order={order_difference:.3g}"
            )
    return rows, maximum_perturbation, maximum_order


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Audit output directory (default: Results/reproducibility).",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--structural-only",
        action="store_true",
        help="Skip the refit-based perturbation and target-order checks.",
    )
    args = parser.parse_args()
    package_root = args.package_root.resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else package_root / "Results" / "reproducibility"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    archived = _load_archived(package_root)
    structural_rows, maximum_actual_difference = _structural_audit(archived)
    behavioral_rows: list[dict[str, Any]] = []
    maximum_perturbation = float("nan")
    maximum_order = float("nan")
    if not args.structural_only:
        behavioral_rows, maximum_perturbation, maximum_order = _behavioral_audit(
            package_root, threads=max(1, int(args.threads))
        )

    origin_audit = pd.DataFrame(structural_rows)
    if behavioral_rows:
        behavioral_frame = pd.DataFrame(behavioral_rows)
        origin_audit = origin_audit.merge(
            behavioral_frame.groupby("configuration", as_index=False).agg(
                audited_behavioral_origins=("origin", "count"),
                maximum_perturbation_difference=(
                    "future_target_perturbation_difference",
                    "max",
                ),
                maximum_order_difference=("target_order_difference", "max"),
                perturbation_verified=("perturbation_pass", "all"),
                target_order_verified=("target_order_pass", "all"),
            ),
            on="configuration",
            how="left",
        )
        behavioral_frame.to_csv(
            output_dir / "behavioral_forecast_audit.csv", index=False
        )
    origin_audit.to_csv(output_dir / "forecast_origin_audit.csv", index=False)

    behavior_verified = bool(behavioral_rows)
    result = {
        "status": "PASS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configurations": 5,
        "origins_per_configuration": 59,
        "hours_per_target": 1416,
        "fit_end_rule_verified": True,
        "history_end_rule_verified": True,
        "future_target_rows_in_fit_set": 0,
        "actual_values_identical_across_configurations": True,
        "maximum_actual_difference": maximum_actual_difference,
        "future_target_perturbation_verified": behavior_verified,
        "target_order_invariance_verified": behavior_verified,
        "maximum_perturbation_difference": (
            maximum_perturbation if behavior_verified else None
        ),
        "maximum_order_difference": maximum_order if behavior_verified else None,
        "behavioral_configurations": list(BEHAVIORAL_CONFIGS),
        "behavioral_origins": [value.isoformat() for value in AUDIT_ORIGINS],
        "tolerance": 1e-10,
        "new_hyperparameter_tuning_performed": False,
    }
    (output_dir / "forecast_leakage_audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, default=_json_default))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build legacy-compatible probabilistic outputs from search evaluations.

The point forecasts are produced by ``evaluate_search.py``. This adapter uses
the same fixed, non-adaptive empirical trajectory calibration implemented in
``proper_model_comparison.py``:

* 92 complete daily P-Q residual trajectories from Q4 2021;
* exact empirical residual ensembles, without bootstrap simulation;
* horizon-specific Q05-Q95 quantiles and finite-sample signed intervals;
* pinball loss, empirical CRPS, coverage, width, and interval scores.

It does not refit a model or alter the point forecasts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

import evaluate_search as evaluator


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import proper_model_comparison as legacy  # noqa: E402


SCRIPT_VERSION = "search-probabilistic-1.1"
CALIBRATION_PERIOD = "q4_2021"
EVALUATION_PERIODS = ("prewar_2022", "late_february_2022")
SNAPSHOTS = (
    "sx_hist",
    "sx_norm_t10",
    "sx_norm_t30",
    "sx_norm_t60",
    "SNaive-24",
    "SNaive-168",
)
DISPLAY_NAMES = {
    "sx_hist": "HIST",
    "sx_norm_t10": "NORM-T10",
    "sx_norm_t30": "NORM-T30",
    "sx_norm_t60": "NORM-T60",
    "SNaive-24": "SNaive-24",
    "SNaive-168": "SNaive-168",
}


def _display_name(snapshot: str) -> str:
    if snapshot in DISPLAY_NAMES:
        return DISPLAY_NAMES[snapshot]
    match = re.fullmatch(r"sx_norm_s(\d+)_t(\d+)", snapshot)
    if match:
        return f"NORM-S{match.group(1)}-T{match.group(2)}"
    return snapshot


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _legacy_point_mapping(
    predictions: pd.DataFrame,
    *,
    snapshot: str,
    periods: Sequence[str],
    expected_origins: int,
) -> Dict[str, pd.DataFrame]:
    block = predictions[
        (predictions["snapshot"] == snapshot)
        & predictions["period"].isin(periods)
        & (predictions["fit_gap_hours"] == 0)
    ].copy()
    if block.empty:
        raise ValueError(f"No predictions for {snapshot}, periods={list(periods)}")
    origin_count = int(block["origin"].nunique())
    if origin_count != int(expected_origins):
        raise ValueError(
            f"{snapshot}: expected {expected_origins} origins, found {origin_count}"
        )

    result: Dict[str, pd.DataFrame] = {}
    for target in evaluator.TARGETS:
        target_block = block[block["target"] == target].copy()
        counts = target_block.groupby("origin")["horizon"].agg(
            ["count", "min", "max", "nunique"]
        )
        invalid = counts[
            (counts["count"] != 24)
            | (counts["min"] != 1)
            | (counts["max"] != 24)
            | (counts["nunique"] != 24)
        ]
        if not invalid.empty:
            raise ValueError(
                f"{snapshot}/{target}: incomplete daily trajectories "
                f"at {invalid.index[:5].tolist()}"
            )
        frame = pd.DataFrame(
            {
                "model_id": _display_name(snapshot),
                "target": target,
                "ts": target_block["timestamp"],
                "date": target_block["origin"].dt.date,
                "horizon": target_block["horizon"].astype(int),
                "train_end": target_block["fit_end"],
                "y_true": target_block["actual"].astype(float),
                "y_pred": target_block["prediction"].astype(float),
            }
        )
        # The legacy interval code defines signed residual as observation minus
        # point forecast. The search evaluator stores the opposite sign.
        frame["error"] = frame["y_true"] - frame["y_pred"]
        frame.sort_values(["ts", "horizon"], inplace=True)
        frame.reset_index(drop=True, inplace=True)
        result[target] = frame
    return result


def _identifiers(snapshot: str) -> tuple[str, str, int, str]:
    if snapshot.startswith("SNaive"):
        lag = snapshot.split("-", maxsplit=1)[1]
        return "baseline", "SeasonalNaive", 0, f"lag{lag}"
    return "normalized_search", "LGBM", 1, "mutual"


def _attach_identifiers(table: pd.DataFrame, snapshot: str) -> None:
    family, model, sheet, lag_policy = _identifiers(snapshot)
    table.insert(1, "family", family)
    table.insert(2, "model", model)
    table.insert(3, "sheet", sheet)
    table.insert(4, "lag_policy", lag_policy)


def _write_model_workbook(
    output_path: Path,
    predictions: Mapping[str, pd.DataFrame],
    daily: pd.DataFrame,
    snapshot: str,
) -> None:
    _, _, sheet, lag_policy = _identifiers(snapshot)
    legacy._write_pair_workbook(
        output_path,
        predictions,
        daily,
        model_name=_display_name(snapshot),
        lag_policy=lag_policy,
        sheet=sheet,
    )


def run(
    package_root: Path,
    *,
    predictions_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    snapshots: Sequence[str] = SNAPSHOTS,
) -> Path:
    package_root = package_root.resolve()
    predictions_path = (
        predictions_path.resolve()
        if predictions_path is not None
        else package_root / "_work" / "forecasting" / "predictions.csv"
    )
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Missing consolidated point predictions: {predictions_path}"
        )
    point = pd.read_csv(
        predictions_path,
        parse_dates=["origin", "timestamp", "fit_end", "history_end"],
    )
    required_periods = {CALIBRATION_PERIOD, *EVALUATION_PERIODS}
    missing_periods = sorted(required_periods - set(point["period"].astype(str)))
    if missing_periods:
        raise ValueError(
            "Probabilistic scoring requires completed calibration and evaluation "
            f"periods; missing={missing_periods}"
        )

    frame = evaluator.core._read_pq_sheet_xlsx(
        package_root / "Input" / "PQ.xlsx",
        evaluator.core.EXPERIMENT_SHEET,
    )
    output_dir = (
        output_dir.resolve()
        if output_dir is not None
        else package_root / "_work" / "probabilistic"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[pd.DataFrame] = []
    calibration_summaries: list[pd.DataFrame] = []
    daily_parts: list[pd.DataFrame] = []
    horizon_parts: list[pd.DataFrame] = []
    predictions_for_dm: Dict[str, Dict[str, pd.DataFrame]] = {
        target: {} for target in evaluator.TARGETS
    }

    for snapshot in snapshots:
        display = _display_name(snapshot)
        calibration = _legacy_point_mapping(
            point,
            snapshot=snapshot,
            periods=(CALIBRATION_PERIOD,),
            expected_origins=92,
        )
        evaluation = _legacy_point_mapping(
            point,
            snapshot=snapshot,
            periods=EVALUATION_PERIODS,
            expected_origins=59,
        )
        probabilistic = legacy._add_probabilistic_predictions(
            evaluation,
            calibration,
            horizon=24,
            adaptive=False,
            stratify_weekend=False,
            minimum_pool_days=12,
            bootstrap_simulations=0,
            random_seed=42,
        )
        calibration_summary, _, _ = legacy._summary_tables(
            frame,
            calibration,
            "2021-10-01 00:00",
            model_id=display,
            period_name="calibration_point",
            mase_reference=frame,
        )
        summary, daily, horizon = legacy._summary_tables(
            frame,
            probabilistic,
            "2022-01-01 00:00",
            model_id=display,
            period_name="evaluation",
            mase_reference=frame,
        )
        for table in (calibration_summary, summary, daily, horizon):
            _attach_identifiers(table, snapshot)

        _write_model_workbook(
            output_dir / f"{display}.xlsx",
            probabilistic,
            daily,
            snapshot,
        )
        calibration_summaries.append(calibration_summary)
        summaries.append(summary)
        daily_parts.append(daily)
        horizon_parts.append(horizon)
        for target in evaluator.TARGETS:
            predictions_for_dm[target][display] = probabilistic[target]
        print(f"[probabilistic] {display}: complete", flush=True)

    calibration_summary = pd.concat(calibration_summaries, ignore_index=True)
    evaluation_summary = legacy._rank_summary(
        pd.concat(summaries, ignore_index=True)
    )
    daily_metrics = pd.concat(daily_parts, ignore_index=True)
    horizon_metrics = pd.concat(horizon_parts, ignore_index=True)
    calibration_summary.to_csv(
        output_dir / "calibration_point_summary.csv", index=False
    )
    evaluation_summary.to_csv(
        output_dir / "evaluation_summary.csv", index=False
    )
    daily_metrics.to_csv(output_dir / "daily_metrics.csv", index=False)
    horizon_metrics.to_csv(output_dir / "horizon_metrics.csv", index=False)

    comparison_path = output_dir / "model_comparison.xlsx"
    with pd.ExcelWriter(comparison_path, engine="openpyxl") as writer:
        evaluation_summary.to_excel(
            writer, sheet_name="evaluation_summary", index=False
        )
        calibration_summary.to_excel(
            writer, sheet_name="calibration_summary", index=False
        )
        daily_metrics.to_excel(writer, sheet_name="daily_metrics", index=False)
        horizon_metrics.to_excel(
            writer, sheet_name="horizon_metrics", index=False
        )
        for target in evaluator.TARGETS:
            short = legacy.TARGET_SHORT[target]
            for loss in ("squared", "absolute", "crps", "pinball"):
                matrices = legacy._pairwise_dm_matrices(
                    predictions_for_dm[target],
                    loss=loss,
                    horizon=24,
                )
                code = {
                    "squared": "SE",
                    "absolute": "AE",
                    "crps": "CRPS",
                    "pinball": "PB",
                }[loss]
                for key, suffix in (
                    ("stat", "stat"),
                    ("p", "p"),
                    ("p_holm", "holm"),
                    ("n", "n"),
                ):
                    matrices[key].to_excel(
                        writer,
                        sheet_name=f"{short}_{code}_DM_{suffix}",
                    )
        pd.DataFrame(
            {
                "item": [
                    "calibration period",
                    "evaluation period",
                    "calibration trajectories",
                    "adaptive calibration",
                    "empirical ensemble",
                    "interval construction",
                    "dependence",
                    "point-model status",
                ],
                "value": [
                    "2021-10-01 through 2021-12-31",
                    "2022-01-01 through 2022-02-28",
                    "92 complete recursive daily P-Q paths",
                    "disabled",
                    "exact; no bootstrap resampling",
                    "signed horizon-specific finite-sample order statistics",
                    "whole paired P-Q daily error paths retained",
                    "fixed snapshots; probabilistic scores do not retune models",
                ],
            }
        ).to_excel(writer, sheet_name="protocol", index=False)

    compact_columns = [
        "model_id",
        "target",
        "RMSE",
        "MAE",
        "WMAPE%",
        "MASE",
        "CRPS",
        "Pinball_mean",
        "Coverage80",
        "Width80",
        "IntervalScore80",
        "Coverage95",
        "Width95",
        "IntervalScore95",
        "CalibrationMAE",
    ]
    evaluation_summary[compact_columns].to_csv(
        output_dir / "probabilistic_metrics_compact.csv", index=False
    )
    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "calibration_period": CALIBRATION_PERIOD,
        "calibration_origins": 92,
        "evaluation_periods": list(EVALUATION_PERIODS),
        "evaluation_origins": 59,
        "snapshots": list(snapshots),
        "quantile_grid": list(legacy.QUANTILE_GRID),
        "interval_coverages": list(legacy.INTERVAL_COVERAGES),
        "adaptive_calibration": False,
        "bootstrap_simulations": 0,
        "point_predictions_sha256": _sha256(predictions_path),
        "files": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(output_dir.iterdir())
            if path.is_file()
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return output_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=PACKAGE_ROOT)
    parser.add_argument(
        "--predictions",
        type=Path,
        help=(
            "Consolidated point-prediction CSV. Defaults to "
            "_work/forecasting/predictions.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory for probabilistic workbooks and tables. Defaults to "
            "_work/probabilistic."
        ),
    )
    parser.add_argument(
        "--snapshots",
        nargs="+",
        default=list(SNAPSHOTS),
        help="Snapshot identifiers to process.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    output = run(
        args.package_root,
        predictions_path=args.predictions,
        output_dir=args.output_dir,
        snapshots=args.snapshots,
    )
    try:
        display_path = output.relative_to(args.package_root)
    except ValueError:
        display_path = output
    print(f"Probabilistic comparison: {display_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Analyze the T60-only tuning-fold breadth experiment.

The inferential comparison is deliberately narrow:

* SEARCH-24, SEARCH-36, and SEARCH-48 use the same LightGBM architecture,
  objective, sampler seed, and 60-terminal-trial budget;
* 1 January through 23 February 2022 (54 daily origins) is the primary period;
* 24-28 February 2022 is reported separately and is not used for selection;
* daily Diebold-Mariano tests cover the three unique T60 breadth pairs only;
* point-loss and CRPS p-values form separate Holm families of size three.

HIST and the two seasonal-naive systems remain descriptive references. Q4 2021
is used for residual-trajectory calibration and is not mixed into the primary
2022 model decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle

import analyze_search as statistics
import evaluate_search as evaluator
import model_registry


SCRIPT_VERSION = "t60-breadth-analysis-1.0"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
POINT_MODELS = (
    "sx_hist",
    "sx_norm_t60",
    "sx_norm_s36_t60",
    "sx_norm_s48_t60",
    "SNaive-24",
    "SNaive-168",
)
T60_MODELS = (
    "sx_norm_t60",
    "sx_norm_s36_t60",
    "sx_norm_s48_t60",
)
MODEL_LABELS = {
    "sx_hist": "HIST",
    "sx_norm_t60": "S24-T60",
    "sx_norm_s36_t60": "S36-T60",
    "sx_norm_s48_t60": "S48-T60",
    "SNaive-24": "SNaive-24",
    "SNaive-168": "SNaive-168",
}
PROBABILISTIC_MODEL_IDS = {
    "HIST": "sx_hist",
    "NORM-T60": "sx_norm_t60",
    "S24-T60": "sx_norm_t60",
    "NORM-S36-T60": "sx_norm_s36_t60",
    "S36-T60": "sx_norm_s36_t60",
    "NORM-S48-T60": "sx_norm_s48_t60",
    "S48-T60": "sx_norm_s48_t60",
    "SNaive-24": "SNaive-24",
    "SNaive-168": "SNaive-168",
}
TARGET_LABELS = {
    "P_Power": "Active power P",
    "Q_Power": "Reactive power Q",
}
TARGET_UNITS = {"P_Power": "kW", "Q_Power": "kVAr"}
COLORS = {
    "sx_hist": "#6B7280",
    "sx_norm_t60": "#2F6B9A",
    "sx_norm_s36_t60": "#D97706",
    "sx_norm_s48_t60": "#198754",
    "SNaive-24": "#B23A48",
    "SNaive-168": "#7A5195",
}
PERIOD_SPECS = {
    "calibration_q4_2021": (("q4_2021",), 92),
    "primary_54d": (("prewar_2022",), 54),
    "descriptive_late_february_5d": (("late_february_2022",), 5),
    "compatibility_january_february_59d": (
        ("prewar_2022", "late_february_2022"),
        59,
    ),
}
PRIMARY_START = pd.Timestamp("2022-01-01")
PRIMARY_END = pd.Timestamp("2022-02-23")
LATE_START = pd.Timestamp("2022-02-24")
LATE_END = pd.Timestamp("2022-02-28")


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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_point_predictions(predictions: pd.DataFrame) -> None:
    required = {
        "snapshot",
        "period",
        "fit_gap_hours",
        "origin",
        "timestamp",
        "horizon",
        "target",
        "actual",
        "prediction",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Point predictions are missing columns: {missing}")
    unknown = sorted(set(predictions["snapshot"].astype(str)) - set(POINT_MODELS))
    absent = sorted(set(POINT_MODELS) - set(predictions["snapshot"].astype(str)))
    if unknown or absent:
        raise ValueError(
            f"Expected exactly the six breadth systems; absent={absent}, "
            f"unexpected={unknown}"
        )
    if set(predictions["fit_gap_hours"].dropna().astype(int)) != {0}:
        raise ValueError("The breadth comparison must contain only gap-0 forecasts")
    key = [
        "snapshot",
        "period",
        "fit_gap_hours",
        "origin",
        "timestamp",
        "target",
    ]
    duplicates = predictions.duplicated(key, keep=False)
    if duplicates.any():
        raise ValueError(
            f"Duplicate point-forecast keys found: {int(duplicates.sum())} rows"
        )
    if predictions[["actual", "prediction"]].isna().any().any():
        raise ValueError("Point predictions contain missing actuals or forecasts")

    for label, (periods, expected_origins) in PERIOD_SPECS.items():
        if len(periods) != 1:
            continue
        period = periods[0]
        block = predictions[predictions["period"] == period]
        for snapshot in POINT_MODELS:
            for target in evaluator.TARGETS:
                item = block[
                    (block["snapshot"] == snapshot)
                    & (block["target"] == target)
                ]
                origins = int(item["origin"].nunique())
                if origins != expected_origins:
                    raise ValueError(
                        f"{label}/{snapshot}/{target}: expected "
                        f"{expected_origins} origins, found {origins}"
                    )
                counts = item.groupby("origin")["horizon"].agg(
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
                        f"{label}/{snapshot}/{target}: incomplete trajectories"
                    )


def _point_metrics(
    predictions: pd.DataFrame,
    *,
    periods: Sequence[str],
    period_label: str,
    expected_origins: int,
    source_frame: pd.DataFrame,
    scales: Mapping[str, float],
) -> pd.DataFrame:
    block = predictions[predictions["period"].isin(periods)].copy()
    rows: list[dict[str, Any]] = []
    for snapshot in POINT_MODELS:
        row: dict[str, Any] = {
            "snapshot": snapshot,
            "system": MODEL_LABELS[snapshot],
            "period": period_label,
            "n_origins": int(
                block.loc[block["snapshot"] == snapshot, "origin"].nunique()
            ),
        }
        if row["n_origins"] != int(expected_origins):
            raise ValueError(
                f"{period_label}/{snapshot}: expected {expected_origins} "
                f"origins, found {row['n_origins']}"
            )
        normalized_rmse: list[float] = []
        for target in evaluator.TARGETS:
            target_block = block[
                (block["snapshot"] == snapshot)
                & (block["target"] == target)
            ]
            error = (
                target_block["prediction"].to_numpy(dtype=float)
                - target_block["actual"].to_numpy(dtype=float)
            )
            actual = target_block["actual"].to_numpy(dtype=float)
            rmse = float(np.sqrt(np.mean(error**2)))
            mae = float(np.mean(np.abs(error)))
            wmape = float(100.0 * np.sum(np.abs(error)) / np.sum(np.abs(actual)))
            period_start = pd.Timestamp(target_block["timestamp"].min())
            mase_scale = evaluator._mase_scale(
                source_frame[target], period_start, lag=24
            )
            prefix = "P" if target == "P_Power" else "Q"
            row[f"{prefix}_RMSE"] = rmse
            row[f"{prefix}_MAE"] = mae
            row[f"{prefix}_WMAPE_percent"] = wmape
            row[f"{prefix}_MASE"] = (
                float(mae / mase_scale)
                if np.isfinite(mase_scale) and mase_scale > 0
                else math.nan
            )
            normalized_rmse.append(rmse / float(scales[target]))
        row["paired_normalized_RMSE"] = float(np.mean(normalized_rmse))
        rows.append(row)
    table = pd.DataFrame(rows)
    table["rank_paired_normalized_RMSE"] = (
        table["paired_normalized_RMSE"]
        .rank(method="min", ascending=True)
        .astype(int)
    )
    order = {snapshot: index for index, snapshot in enumerate(POINT_MODELS)}
    table["_order"] = table["snapshot"].map(order)
    return table.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def _validate_daily_point_losses(daily: pd.DataFrame) -> None:
    required = {
        "snapshot",
        "period",
        "fit_gap_hours",
        "origin",
        "paired_normalized_squared_loss",
    }
    missing = sorted(required - set(daily.columns))
    if missing:
        raise ValueError(f"Daily point losses are missing columns: {missing}")
    primary = daily[
        (daily["period"] == "prewar_2022")
        & (daily["fit_gap_hours"] == 0)
        & daily["snapshot"].isin(T60_MODELS)
    ]
    counts = primary.groupby("snapshot")["origin"].nunique()
    if set(counts.index) != set(T60_MODELS) or not (counts == 54).all():
        raise ValueError(
            "Primary point-loss table must contain 54 paired trajectories "
            f"for each T60 model; counts={counts.to_dict()}"
        )


def _dm_family(
    losses: pd.DataFrame,
    *,
    loss_name: str,
    model_column: str = "snapshot",
    date_column: str = "origin",
    value_column: str = "loss",
) -> pd.DataFrame:
    pivot = losses.pivot(
        index=date_column, columns=model_column, values=value_column
    ).sort_index()
    missing = sorted(set(T60_MODELS) - set(pivot.columns))
    if missing:
        raise ValueError(f"{loss_name}: missing T60 models {missing}")
    pivot = pivot[list(T60_MODELS)].dropna()
    if len(pivot) != 54:
        raise ValueError(
            f"{loss_name}: expected 54 complete paired days, found {len(pivot)}"
        )
    rows: list[dict[str, Any]] = []
    for model_a, model_b in combinations(T60_MODELS, 2):
        result = statistics.dm_test_losses(
            pivot[model_a].to_numpy(dtype=float),
            pivot[model_b].to_numpy(dtype=float),
            hac_lag="auto",
            min_days=20,
        )
        rows.append(
            {
                "loss_family": loss_name,
                "model_a": model_a,
                "system_a": MODEL_LABELS[model_a],
                "model_b": model_b,
                "system_b": MODEL_LABELS[model_b],
                "mean_loss_a": float(pivot[model_a].mean()),
                "mean_loss_b": float(pivot[model_b].mean()),
                **asdict(result),
            }
        )
    table = pd.DataFrame(rows)
    if len(table) != 3:
        raise AssertionError("The T60 breadth family must contain three pairs")
    table["holm_p"] = statistics.holm_adjust(
        table["p_value"].to_numpy(dtype=float)
    )
    table["holm_family_size"] = 3
    table["significant_0_05"] = table["holm_p"] < 0.05
    table["lower_loss_system"] = np.where(
        table["mean_difference"] < 0,
        table["system_a"],
        np.where(table["mean_difference"] > 0, table["system_b"], "tie"),
    )
    return table


def _point_dm_family(daily: pd.DataFrame) -> pd.DataFrame:
    block = daily[
        (daily["period"] == "prewar_2022")
        & (daily["fit_gap_hours"] == 0)
        & daily["snapshot"].isin(T60_MODELS)
    ][["snapshot", "origin", "paired_normalized_squared_loss"]].rename(
        columns={"paired_normalized_squared_loss": "loss"}
    )
    return _dm_family(
        block,
        loss_name="paired normalized daily squared error",
    )


def _prepare_probabilistic_daily(
    daily: pd.DataFrame,
    scales: Mapping[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "model_id",
        "target",
        "date",
        "CRPS",
        "Pinball_mean",
        "Coverage80",
        "Width80",
        "IntervalScore80",
        "Coverage95",
        "Width95",
        "IntervalScore95",
        "CalibrationMAE",
    }
    missing = sorted(required - set(daily.columns))
    if missing:
        raise ValueError(f"Probabilistic daily metrics are missing: {missing}")
    daily = daily.copy()
    daily["snapshot"] = daily["model_id"].map(PROBABILISTIC_MODEL_IDS)
    unmapped = sorted(
        daily.loc[daily["snapshot"].isna(), "model_id"].astype(str).unique()
    )
    if unmapped:
        raise ValueError(f"Unrecognized probabilistic model identifiers: {unmapped}")
    if set(daily["snapshot"].astype(str)) != set(POINT_MODELS):
        raise ValueError("Probabilistic outputs do not contain exactly six systems")
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()

    primary = daily[
        daily["date"].between(PRIMARY_START, PRIMARY_END)
        & daily["snapshot"].isin(T60_MODELS)
    ]
    counts = primary.groupby(["snapshot", "target"])["date"].nunique()
    if len(counts) != 6 or not (counts == 54).all():
        raise ValueError(
            "Primary probabilistic table must contain 54 days for each "
            f"T60 model-target pair; counts={counts.to_dict()}"
        )

    crps = primary.pivot_table(
        index=["snapshot", "date"],
        columns="target",
        values="CRPS",
        aggfunc="first",
    ).reset_index()
    crps["loss"] = 0.5 * (
        crps["P_Power"] / float(scales["P_Power"])
        + crps["Q_Power"] / float(scales["Q_Power"])
    )
    return daily, crps[["snapshot", "date", "loss"]]


def _probabilistic_summary(
    daily: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    period_label: str,
    expected_days: int,
) -> pd.DataFrame:
    block = daily[daily["date"].between(start, end)].copy()
    columns = [
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
    summary = (
        block.groupby(["snapshot", "target"], as_index=False)
        .agg(n_days=("date", "nunique"), **{column: (column, "mean") for column in columns})
    )
    if len(summary) != len(POINT_MODELS) * len(evaluator.TARGETS):
        raise ValueError(f"{period_label}: incomplete probabilistic summary")
    if not (summary["n_days"] == int(expected_days)).all():
        raise ValueError(
            f"{period_label}: expected {expected_days} days per model-target"
        )
    summary.insert(1, "system", summary["snapshot"].map(MODEL_LABELS))
    summary.insert(2, "period", period_label)
    order = {snapshot: index for index, snapshot in enumerate(POINT_MODELS)}
    target_order = {target: index for index, target in enumerate(evaluator.TARGETS)}
    summary["_model_order"] = summary["snapshot"].map(order)
    summary["_target_order"] = summary["target"].map(target_order)
    return (
        summary.sort_values(["_model_order", "_target_order"])
        .drop(columns=["_model_order", "_target_order"])
        .reset_index(drop=True)
    )


def _search_tables(package_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    jobs = (
        ("SEARCH-24", "sx_norm_t60"),
        ("SEARCH-36", "sx_norm_s36_t60"),
        ("SEARCH-48", "sx_norm_s48_t60"),
    )
    completion_rows: list[dict[str, Any]] = []
    for design, snapshot in jobs:
        cost_path = package_root / "_work" / "search" / snapshot / "cost.json"
        meta_path = model_registry.metadata_path(
            package_root / "Models", snapshot, "P_Power"
        )
        cost = _read_json(cost_path)
        metadata = _read_json(meta_path)
        source = metadata["source"]
        params = metadata["tuned_params"]
        completion_rows.append(
            {
                "design": design,
                "snapshot": snapshot,
                "system": MODEL_LABELS[snapshot],
                "folds": int(source["trial_target_fold_fits"]) // 2,
                "terminal_trials": int(cost["terminal_trials"]),
                "complete_trials": int(cost["complete_trials"]),
                "pruned_trials": int(cost["pruned_trials"]),
                "actual_target_fold_fits": int(cost["actual_target_fold_fits"]),
                "trial_duration_seconds": float(cost["trial_duration_seconds"]),
                "selected_trial": int(source["trial_number"]),
                "selected_objective": float(source["objective"]),
                "n_estimators": int(params["n_estimators"]),
                "learning_rate": float(params["learning_rate"]),
                "num_leaves": int(params["num_leaves"]),
                "max_depth": int(params["max_depth"]),
                "min_child_samples": int(params["min_child_samples"]),
                "subsample": float(params["subsample"]),
                "colsample_bytree": float(params["colsample_bytree"]),
                "reg_alpha": float(params["reg_alpha"]),
                "reg_lambda": float(params["reg_lambda"]),
            }
        )
    completion = pd.DataFrame(completion_rows)

    rescoring_rows: list[dict[str, Any]] = []
    design_jobs = (
        ("SEARCH-36", "sx_norm_s36_t60"),
        ("SEARCH-48", "sx_norm_s48_t60"),
    )
    anchor_labels = {
        "historical_MUT_L24": "HIST",
        "standard_LGBM_default": "LightGBM default",
        "completed_sx_norm_t60": "S24-T60",
        "completed_sx_norm_s36_t60": "S36-T60",
    }
    for design, snapshot in design_jobs:
        trial_path = (
            package_root / "_work" / "search" / snapshot / "trials.csv"
        )
        trials = pd.read_csv(trial_path)
        meta = _read_json(
            model_registry.metadata_path(
                package_root / "Models", snapshot, "P_Power"
            )
        )
        selected_trial = int(meta["source"]["trial_number"])
        selected = trials[trials["trial_number"] == selected_trial]
        if len(selected) != 1:
            raise ValueError(f"{snapshot}: selected trial is not unique")
        anchors = trials[
            (trials["state"] == "COMPLETE")
            & trials["anchor_name"].notna()
            & (trials["anchor_name"].astype(str) != "")
        ].copy()
        candidates: list[tuple[pd.Series, str, str]] = []
        for _, item in anchors.iterrows():
            candidates.append(
                (
                    item,
                    anchor_labels.get(str(item["anchor_name"]), str(item["anchor_name"])),
                    "inherited anchor",
                )
            )
        candidates.append((selected.iloc[0], MODEL_LABELS[snapshot], "selected"))
        best_inherited = float(anchors["objective"].min())
        for item, label, role in candidates:
            objective = float(item["objective"])
            rescoring_rows.append(
                {
                    "design": design,
                    "candidate": label,
                    "role": role,
                    "trial_number": int(item["trial_number"]),
                    "objective": objective,
                    "P_RMSE": float(item["P_RMSE"]),
                    "Q_RMSE": float(item["Q_RMSE"]),
                    "best_inherited_objective": best_inherited,
                    "percent_vs_best_inherited": float(
                        100.0 * (objective - best_inherited) / best_inherited
                    ),
                    "parameter_hash": str(item["parameter_hash"]),
                }
            )
    rescoring = pd.DataFrame(rescoring_rows)
    return completion, rescoring


def _save_figure(figure: plt.Figure, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(base_path.with_suffix(".png"), dpi=320, bbox_inches="tight")
    figure.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_pq_rmse(metrics: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    zoom_models = ("sx_hist", *T60_MODELS)
    marker_by_snapshot = {
        snapshot: (
            "o" if snapshot in T60_MODELS else (
                "s" if snapshot == "sx_hist" else "^"
            )
        )
        for snapshot in POINT_MODELS
    }
    for axis, models, title in (
        (axes[0], POINT_MODELS, "(a) All retained systems"),
        (axes[1], zoom_models, "(b) LightGBM detail"),
    ):
        for snapshot in models:
            row = metrics[metrics["snapshot"] == snapshot].iloc[0]
            axis.scatter(
                row["P_RMSE"],
                row["Q_RMSE"],
                s=78 if snapshot in T60_MODELS else 62,
                marker=marker_by_snapshot[snapshot],
                color=COLORS[snapshot],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
                label=MODEL_LABELS[snapshot],
            )
        axis.set_xlabel("Active-power RMSE (kW)")
        axis.set_ylabel("Reactive-power RMSE (kVAr)")
        axis.set_title(title)
        axis.grid(True, color="#D9D9D9", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].add_patch(
        Rectangle(
            (116.0, 113.8),
            11.0,
            8.2,
            fill=False,
            edgecolor="#444444",
            linewidth=0.9,
            linestyle=":",
        )
    )
    axes[0].annotate(
        "detail in (b)",
        (127.0, 122.0),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
    )
    axes[1].set_xlim(116.0, 127.0)
    axes[1].set_ylim(113.8, 122.0)
    label_offsets = {
        "sx_hist": (7, 5),
        "sx_norm_t60": (7, 7),
        "sx_norm_s36_t60": (-38, -18),
        "sx_norm_s48_t60": (7, -18),
    }
    for snapshot in zoom_models:
        row = metrics[metrics["snapshot"] == snapshot].iloc[0]
        axes[1].annotate(
            MODEL_LABELS[snapshot],
            (row["P_RMSE"], row["Q_RMSE"]),
            xytext=label_offsets[snapshot],
            textcoords="offset points",
            fontsize=8.2,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.05),
    )
    figure.suptitle("Primary 54-origin point-forecast comparison", y=1.01)
    figure.tight_layout(rect=(0, 0.07, 1, 0.97))
    _save_figure(figure, path)


def _plot_horizon_rmse(predictions: pd.DataFrame, path: Path) -> None:
    block = predictions[predictions["period"] == "prewar_2022"].copy()
    block["squared_error"] = (block["prediction"] - block["actual"]) ** 2
    horizon = (
        block.groupby(["snapshot", "target", "horizon"], as_index=False)
        .agg(RMSE=("squared_error", lambda values: float(np.sqrt(np.mean(values)))))
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), sharex=True)
    for axis, target in zip(axes, evaluator.TARGETS):
        target_block = horizon[horizon["target"] == target]
        for snapshot in POINT_MODELS:
            item = target_block[target_block["snapshot"] == snapshot]
            axis.plot(
                item["horizon"],
                item["RMSE"],
                color=COLORS[snapshot],
                linewidth=1.8 if snapshot in T60_MODELS else 1.2,
                linestyle="--" if snapshot.startswith("SNaive") else "-",
                label=MODEL_LABELS[snapshot],
            )
        axis.set_title(TARGET_LABELS[target])
        axis.set_xlabel("Forecast horizon (h)")
        axis.set_ylabel(f"RMSE ({TARGET_UNITS[target]})")
        axis.set_xticks([1, 6, 12, 18, 24])
        axis.grid(True, color="#DDDDDD", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )
    figure.suptitle("Primary 54-origin error by forecast horizon", y=1.01)
    figure.tight_layout()
    _save_figure(figure, path)


def _plot_trajectories(
    predictions: pd.DataFrame,
    *,
    period: str,
    title: str,
    path: Path,
) -> None:
    block = predictions[predictions["period"] == period].copy()
    figure, axes = plt.subplots(2, 1, figsize=(11.2, 5.7), sharex=True)
    for axis, target in zip(axes, evaluator.TARGETS):
        target_block = block[block["target"] == target]
        actual = (
            target_block[["timestamp", "actual"]]
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
        )
        axis.plot(
            actual["timestamp"],
            actual["actual"],
            color="#222222",
            linewidth=1.1,
            label="Observed",
            zorder=4,
        )
        for snapshot in T60_MODELS:
            item = target_block[target_block["snapshot"] == snapshot].sort_values(
                "timestamp"
            )
            axis.plot(
                item["timestamp"],
                item["prediction"],
                color=COLORS[snapshot],
                linewidth=0.9,
                alpha=0.9,
                label=MODEL_LABELS[snapshot],
            )
        axis.set_ylabel(f"{TARGET_LABELS[target]} ({TARGET_UNITS[target]})")
        axis.grid(True, color="#E0E0E0", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    axes[-1].set_xlabel("Forecast timestamp")
    figure.suptitle(title, y=0.995)
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))
    _save_figure(figure, path)


def _dm_matrix(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    size = len(T60_MODELS)
    p_values = np.full((size, size), np.nan, dtype=float)
    directions = np.zeros((size, size), dtype=float)
    index = {model: position for position, model in enumerate(T60_MODELS)}
    for _, row in table.iterrows():
        first = index[str(row["model_a"])]
        second = index[str(row["model_b"])]
        p_value = float(row["holm_p"])
        mean = float(row["mean_difference"])
        p_values[first, second] = p_values[second, first] = p_value
        directions[first, second] = -np.sign(mean)
        directions[second, first] = np.sign(mean)
    return p_values, directions


def _plot_dm_matrices(
    point_dm: pd.DataFrame,
    crps_dm: pd.DataFrame,
    path: Path,
) -> None:
    labels = [MODEL_LABELS[model] for model in T60_MODELS]
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 4.0))
    for axis, table, title in (
        (axes[0], point_dm, "(a) Paired normalized squared error"),
        (axes[1], crps_dm, "(b) Paired normalized CRPS"),
    ):
        p_values, directions = _dm_matrix(table)
        color = np.zeros((3, 3), dtype=float)
        for row in range(3):
            for column in range(3):
                if row == column:
                    color[row, column] = 0.0
                elif np.isfinite(p_values[row, column]):
                    multiplier = 2.0 if p_values[row, column] < 0.05 else 1.0
                    color[row, column] = multiplier * directions[row, column]
        annotations = np.empty((3, 3), dtype=object)
        for row in range(3):
            for column in range(3):
                if row == column:
                    annotations[row, column] = "-"
                else:
                    value = p_values[row, column]
                    annotations[row, column] = (
                        "<0.001" if value < 0.001 else f"{value:.3f}"
                    )
        sns.heatmap(
            color,
            ax=axis,
            cmap=ListedColormap(
                ["#B23A48", "#F4DADA", "#D9D9D9", "#DDEFE3", "#2E7D4F"]
            ),
            norm=BoundaryNorm(
                [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5],
                ncolors=5,
            ),
            annot=annotations,
            fmt="",
            cbar=False,
            linewidths=1,
            linecolor="white",
            square=True,
            xticklabels=labels,
            yticklabels=labels,
            annot_kws={"fontsize": 9},
        )
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Column system")
        axis.set_ylabel("Row system")
        axis.tick_params(axis="x", rotation=35)
        axis.tick_params(axis="y", rotation=0)
    figure.suptitle(
        "Holm-adjusted p-values for the three T60 breadth comparisons\n"
        "pale tint: direction of mean loss only; dark tint: significant at 5%",
        fontsize=10.5,
        y=1.05,
    )
    figure.tight_layout()
    _save_figure(figure, path)


def _plot_probabilistic_scores(summary: pd.DataFrame, path: Path) -> None:
    block = summary[summary["snapshot"].isin(POINT_MODELS)].copy()
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for axis, target in zip(axes, evaluator.TARGETS):
        item = block[block["target"] == target]
        x = np.arange(len(item))
        axis.bar(
            x,
            item["CRPS"],
            color=[COLORS[snapshot] for snapshot in item["snapshot"]],
            width=0.72,
        )
        for position, value in zip(x, item["CRPS"]):
            axis.annotate(
                f"{value:.2f}",
                (position, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
        axis.set_xticks(x, item["system"], rotation=35, ha="right")
        axis.set_ylabel(f"CRPS ({TARGET_UNITS[target]})")
        axis.set_title(TARGET_LABELS[target])
        axis.grid(True, axis="y", color="#DDDDDD", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Primary 54-origin probabilistic accuracy", y=1.01)
    figure.tight_layout()
    _save_figure(figure, path)


def _plot_search_rescoring(rescoring: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.2, 3.8),
        layout="constrained",
    )
    for column, design in enumerate(("SEARCH-36", "SEARCH-48")):
        item = rescoring[rescoring["design"] == design].copy()
        axis = axes[column]
        y = np.arange(len(item))
        colors = [
            "#198754" if role == "selected" else "#9CA3AF"
            for role in item["role"]
        ]
        axis.barh(
            y,
            item["percent_vs_best_inherited"],
            color=colors,
            height=0.62,
        )
        axis.scatter(
            item["percent_vs_best_inherited"],
            y,
            color=colors,
            edgecolor="white",
            linewidth=0.6,
            s=24,
            zorder=3,
        )
        for position, value in zip(y, item["percent_vs_best_inherited"]):
            offset = 5 if value < 0.5 else 4
            axis.annotate(
                f"{value:+.4f}%",
                (value, position),
                xytext=(offset, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8,
            )
        axis.axvline(0.0, color="#333333", linewidth=0.8)
        axis.set_yticks(y, item["candidate"])
        axis.invert_yaxis()
        axis.set_xlabel("Difference from best inherited objective (%)")
        axis.set_title(design)
        upper = max(1.0, float(item["percent_vs_best_inherited"].max()) * 1.18)
        axis.set_xlim(-0.8, upper)
        axis.grid(True, axis="x", color="#E0E0E0", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=8.5)
    figure.suptitle(
        "Within-design objective relative to the best inherited anchor "
        "(lower is better)",
    )
    _save_figure(figure, path)


def _plot_nested_split_breadth(package_root: Path, path: Path) -> None:
    specifications = (
        ("SEARCH-24", "splits_search24.xlsx"),
        ("SEARCH-36", "splits_search36.xlsx"),
        ("SEARCH-48", "splits_search48.xlsx"),
    )
    layer_colors = {
        "CORE24": "#2F6B9A",
        "EXPANSION_A": "#D97706",
        "EXPANSION_B": "#198754",
    }
    stratum_markers = {"regular": "o", "calendar": "s", "stress": "^"}
    figure, axis = plt.subplots(figsize=(10.2, 3.5))
    for y_position, (design, filename) in enumerate(specifications):
        frame = pd.read_excel(
            package_root / "Input" / filename,
            sheet_name="24",
        )
        frame["test_start"] = pd.to_datetime(frame["test_start"])
        if "origin_layer" not in frame:
            frame["origin_layer"] = "CORE24"
        for (layer, stratum), group in frame.groupby(
            ["origin_layer", "stratum"], sort=False
        ):
            axis.scatter(
                group["test_start"],
                np.full(len(group), y_position),
                color=layer_colors[str(layer)],
                marker=stratum_markers[str(stratum)],
                s=42,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
    axis.set_yticks(range(len(specifications)), [item[0] for item in specifications])
    axis.invert_yaxis()
    axis.set_xlabel("Scored 24-hour tuning origin in 2021")
    axis.set_title("Nested chronological tuning-fold breadth")
    axis.grid(True, axis="x", color="#DDDDDD", linewidth=0.6)
    axis.spines[["top", "right", "left"]].set_visible(False)
    layer_handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=color,
            label=layer.replace("_", " ").title(),
            markersize=7,
        )
        for layer, color in layer_colors.items()
    ]
    stratum_handles = [
        plt.Line2D(
            [],
            [],
            marker=marker,
            linestyle="",
            color="#555555",
            label=stratum.title(),
            markersize=7,
        )
        for stratum, marker in stratum_markers.items()
    ]
    first_legend = axis.legend(
        handles=layer_handles,
        loc="upper center",
        bbox_to_anchor=(0.34, -0.2),
        ncol=3,
        frameon=False,
    )
    axis.add_artist(first_legend)
    axis.legend(
        handles=stratum_handles,
        loc="upper center",
        bbox_to_anchor=(0.78, -0.2),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout()
    _save_figure(figure, path)


def _write_report(
    path: Path,
    *,
    primary_metrics: pd.DataFrame,
    primary_probabilistic: pd.DataFrame,
    point_dm: pd.DataFrame,
    crps_dm: pd.DataFrame,
    rescoring: pd.DataFrame,
    scales: Mapping[str, float],
) -> None:
    point_leader = primary_metrics.sort_values("paired_normalized_RMSE").iloc[0]
    t60_point = primary_metrics[
        primary_metrics["snapshot"].isin(T60_MODELS)
    ].sort_values("paired_normalized_RMSE")
    prob_t60 = primary_probabilistic[
        primary_probabilistic["snapshot"].isin(T60_MODELS)
    ]
    paired_crps = (
        prob_t60.pivot(index="snapshot", columns="target", values="CRPS")
        .assign(
            paired=lambda table: 0.5
            * (
                table["P_Power"] / float(scales["P_Power"])
                + table["Q_Power"] / float(scales["Q_Power"])
            )
        )
        .sort_values("paired")
    )
    point_significant = int(point_dm["significant_0_05"].sum())
    crps_significant = int(crps_dm["significant_0_05"].sum())
    s36_selected = rescoring[
        (rescoring["design"] == "SEARCH-36") & (rescoring["role"] == "selected")
    ].iloc[0]
    s48_selected = rescoring[
        (rescoring["design"] == "SEARCH-48") & (rescoring["role"] == "selected")
    ].iloc[0]
    text = f"""# T60 Tuning-Fold Breadth Experiment

## Protocol

SEARCH-24, SEARCH-36, and SEARCH-48 use the same LightGBM forecasting
architecture, fixed normalized objective scales, TPE seed, pruning policy, and
budget of 60 nonduplicate terminal trials. The wider designs add nested tuning
origins; they do not change the model inputs or recursive forecasting rule.
This is a fixed-trial comparison, not an equal-compute comparison: evaluating
one candidate on 36 or 48 origins requires more target-fold fits than on 24.

The primary comparison contains 54 midnight origins from 1 January through
23 February 2022. The five origins from 24-28 February are descriptive only.
Q4 2021 supplies fixed, model-specific residual trajectories for probabilistic
calibration and does not enter the primary model decision. It is out of the
new SEARCH-24/36/48 schedules; for HIST it remains descriptive because the
archived historical tuning schedule includes later 2021 folds.

## Development Search

On SEARCH-36, the selected S36-T60 vector changes the normalized objective by
{s36_selected['percent_vs_best_inherited']:.6f}% relative to the best inherited
anchor evaluated on the same 36 origins. On SEARCH-48, S48-T60 changes it by
{s48_selected['percent_vs_best_inherited']:.6f}% relative to the best inherited
anchor evaluated on the same 48 origins. Objectives from different split
designs are not compared as though they were measured on one common sample.

## Primary 2022 Result

The numerical point leader across all six systems is
{point_leader['system']} (paired normalized RMSE
{point_leader['paired_normalized_RMSE']:.6f}). Among the three T60 systems,
the ordering is {', '.join(t60_point['system'].tolist())}.

The paired normalized CRPS ordering among the T60 systems is
{', '.join(MODEL_LABELS[index] for index in paired_crps.index)}.

The three point-loss contrasts form one Holm family and the three CRPS
contrasts form a separate Holm family. At the 5% family-wise level,
{point_significant} of 3 point-loss contrasts and {crps_significant} of 3 CRPS
contrasts are significant. Numerical leadership is therefore reported
separately from evidence of a statistically resolved difference.

## Interpretation

The experiment tests whether increasing the breadth of the chronological tuning
sample changes the retained T60 specification and improves later forecasts. It
does not test T10 versus T30 versus T60, and it does not justify selecting a
model from the final five February origins. HIST and the seasonal-naive systems
are references; the predefined Holm families contain only S24-T60, S36-T60,
and S48-T60.
"""
    path.write_text(text, encoding="utf-8")


def run(
    package_root: Path,
    *,
    point_predictions_path: Optional[Path] = None,
    point_daily_path: Optional[Path] = None,
    probabilistic_daily_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    package_root = package_root.resolve()
    point_root = package_root / "_work" / "breadth_t60" / "point"
    point_predictions_path = (
        point_predictions_path.resolve()
        if point_predictions_path is not None
        else point_root / "predictions.csv"
    )
    point_daily_path = (
        point_daily_path.resolve()
        if point_daily_path is not None
        else point_root / "daily_paired_losses.csv"
    )
    probabilistic_daily_path = (
        probabilistic_daily_path.resolve()
        if probabilistic_daily_path is not None
        else package_root
        / "_work"
        / "breadth_t60"
        / "probabilistic"
        / "daily_metrics.csv"
    )
    output_dir = (
        output_dir.resolve()
        if output_dir is not None
        else package_root / "_work" / "breadth_t60" / "analysis"
    )
    for path in (
        point_predictions_path,
        point_daily_path,
        probabilistic_daily_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    predictions = pd.read_csv(
        point_predictions_path,
        parse_dates=["origin", "timestamp", "fit_end", "history_end"],
    )
    daily_point = pd.read_csv(point_daily_path, parse_dates=["origin"])
    daily_probabilistic = pd.read_csv(probabilistic_daily_path)
    _validate_point_predictions(predictions)
    _validate_daily_point_losses(daily_point)

    source_frame = evaluator.core._read_pq_sheet_xlsx(
        package_root / "Input" / "PQ.xlsx",
        evaluator.core.EXPERIMENT_SHEET,
    )
    scales_json = _read_json(package_root / "Input" / "objective_scales.json")
    scales = {
        target: float(scales_json["target_scales"][target])
        for target in evaluator.TARGETS
    }
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    statistics_dir = output_dir / "statistics"
    for directory in (tables_dir, figures_dir, statistics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    point_tables: dict[str, pd.DataFrame] = {}
    for label, (periods, expected_origins) in PERIOD_SPECS.items():
        table = _point_metrics(
            predictions,
            periods=periods,
            period_label=label,
            expected_origins=expected_origins,
            source_frame=source_frame,
            scales=scales,
        )
        point_tables[label] = table
        table.to_csv(tables_dir / f"point_metrics_{label}.csv", index=False)

    probabilistic_daily, crps_losses = _prepare_probabilistic_daily(
        daily_probabilistic, scales
    )
    probabilistic_tables = {
        "primary_54d": _probabilistic_summary(
            probabilistic_daily,
            start=PRIMARY_START,
            end=PRIMARY_END,
            period_label="primary_54d",
            expected_days=54,
        ),
        "descriptive_late_february_5d": _probabilistic_summary(
            probabilistic_daily,
            start=LATE_START,
            end=LATE_END,
            period_label="descriptive_late_february_5d",
            expected_days=5,
        ),
        "compatibility_january_february_59d": _probabilistic_summary(
            probabilistic_daily,
            start=PRIMARY_START,
            end=LATE_END,
            period_label="compatibility_january_february_59d",
            expected_days=59,
        ),
    }
    for label, table in probabilistic_tables.items():
        table.to_csv(
            tables_dir / f"probabilistic_metrics_{label}.csv", index=False
        )

    point_dm = _point_dm_family(daily_point)
    crps_dm = _dm_family(
        crps_losses,
        loss_name="paired normalized daily CRPS",
        date_column="date",
    )
    point_dm.to_csv(statistics_dir / "dm_point_t60_primary_54d.csv", index=False)
    crps_dm.to_csv(statistics_dir / "dm_crps_t60_primary_54d.csv", index=False)

    completion, rescoring = _search_tables(package_root)
    completion.to_csv(tables_dir / "search_completion.csv", index=False)
    rescoring.to_csv(tables_dir / "within_design_anchor_rescoring.csv", index=False)

    workbook_path = output_dir / "t60_breadth_comparison.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        point_tables["primary_54d"].to_excel(
            writer, sheet_name="point_primary_54d", index=False
        )
        point_tables["descriptive_late_february_5d"].to_excel(
            writer, sheet_name="point_late_feb_5d", index=False
        )
        point_tables["calibration_q4_2021"].to_excel(
            writer, sheet_name="point_q4_calibration", index=False
        )
        point_tables["compatibility_january_february_59d"].to_excel(
            writer, sheet_name="point_jan_feb_59d", index=False
        )
        probabilistic_tables["primary_54d"].to_excel(
            writer, sheet_name="prob_primary_54d", index=False
        )
        probabilistic_tables["descriptive_late_february_5d"].to_excel(
            writer, sheet_name="prob_late_feb_5d", index=False
        )
        probabilistic_tables["compatibility_january_february_59d"].to_excel(
            writer, sheet_name="prob_jan_feb_59d", index=False
        )
        point_dm.to_excel(writer, sheet_name="DM_point_family3", index=False)
        crps_dm.to_excel(writer, sheet_name="DM_CRPS_family3", index=False)
        completion.to_excel(writer, sheet_name="search_completion", index=False)
        rescoring.to_excel(writer, sheet_name="anchor_rescoring", index=False)
        pd.DataFrame(
            {
                "item": [
                    "primary period",
                    "descriptive stress period",
                    "calibration period",
                    "T60 systems in paired test families",
                    "point DM loss",
                    "point Holm family size",
                    "probabilistic DM loss",
                    "probabilistic Holm family size",
                    "HIST and seasonal-naive role",
                ],
                "value": [
                    "2022-01-01 through 2022-02-23; 54 complete daily paths",
                    "2022-02-24 through 2022-02-28; five paths; no selection",
                    "Q4 2021; 92 model-specific residual trajectories",
                    "S24-T60, S36-T60, S48-T60",
                    "paired normalized daily squared error",
                    3,
                    "paired normalized daily CRPS",
                    3,
                    "descriptive references; excluded from Holm families",
                ],
            }
        ).to_excel(writer, sheet_name="protocol", index=False)

    _plot_pq_rmse(
        point_tables["primary_54d"],
        figures_dir / "primary_54d_pq_rmse",
    )
    _plot_horizon_rmse(
        predictions,
        figures_dir / "primary_54d_horizon_rmse",
    )
    _plot_trajectories(
        predictions,
        period="prewar_2022",
        title="Primary 54-origin recursive forecasts",
        path=figures_dir / "primary_54d_forecast_trajectories",
    )
    _plot_trajectories(
        predictions,
        period="late_february_2022",
        title="Descriptive forecasts, 24-28 February 2022",
        path=figures_dir / "descriptive_late_february_5d_trajectories",
    )
    _plot_dm_matrices(
        point_dm,
        crps_dm,
        figures_dir / "t60_holm_matrices",
    )
    _plot_probabilistic_scores(
        probabilistic_tables["primary_54d"],
        figures_dir / "primary_54d_probabilistic_crps",
    )
    _plot_search_rescoring(
        rescoring,
        figures_dir / "within_design_anchor_rescoring",
    )
    _plot_nested_split_breadth(
        package_root,
        figures_dir / "nested_split_breadth",
    )

    _write_report(
        output_dir / "README.md",
        primary_metrics=point_tables["primary_54d"],
        primary_probabilistic=probabilistic_tables["primary_54d"],
        point_dm=point_dm,
        crps_dm=crps_dm,
        rescoring=rescoring,
        scales=scales,
    )
    generated_files = sorted(
        path for path in output_dir.rglob("*") if path.is_file()
    )
    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "primary_period": {
            "start": str(PRIMARY_START.date()),
            "end": str(PRIMARY_END.date()),
            "origins": 54,
        },
        "descriptive_period": {
            "start": str(LATE_START.date()),
            "end": str(LATE_END.date()),
            "origins": 5,
        },
        "calibration_period": {
            "start": "2021-10-01",
            "end": "2021-12-31",
            "origins": 92,
        },
        "tested_models": list(T60_MODELS),
        "reference_models": ["sx_hist", "SNaive-24", "SNaive-168"],
        "holm_families": {
            "point": {
                "loss": "paired normalized daily squared error",
                "pairs": 3,
            },
            "probabilistic": {
                "loss": "paired normalized daily CRPS",
                "pairs": 3,
            },
        },
        "objective_scales": scales,
        "inputs": {
            str(path.resolve()): _sha256(path)
            for path in (
                point_predictions_path,
                point_daily_path,
                probabilistic_daily_path,
            )
        },
        "files": {
            str(path.relative_to(output_dir)): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in generated_files
            if path.name != "manifest.json"
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return output_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=PACKAGE_ROOT)
    parser.add_argument("--point-predictions", type=Path)
    parser.add_argument("--point-daily", type=Path)
    parser.add_argument("--probabilistic-daily", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = _build_parser().parse_args(argv)
    output = run(
        args.package_root,
        point_predictions_path=args.point_predictions,
        point_daily_path=args.point_daily,
        probabilistic_daily_path=args.probabilistic_daily,
        output_dir=args.output_dir,
    )
    print(f"T60 breadth analysis: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

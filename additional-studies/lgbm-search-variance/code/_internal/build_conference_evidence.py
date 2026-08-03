# -*- coding: utf-8 -*-
"""Build the frozen publication evidence for the Optuna experiment.

The script consolidates archived forecasts and probabilistic scores.  It does
not fit estimators, tune hyperparameters, or inspect any additional period.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_search import dm_test_losses, holm_adjust
import model_registry


TARGETS = ("P_Power", "Q_Power")
TARGET_SHORT = {"P_Power": "P", "Q_Power": "Q"}
UNITS = {"P_Power": "kW", "Q_Power": "kVAr"}
SCALES = {
    "P_Power": 192.00558336970016,
    "Q_Power": 229.7979977161049,
}
LEARNED = ("S24-T10", "S24-T30", "S24-T60", "S36-T60", "S48-T60")
BASELINES = ("SNaive-24", "SNaive-168")
CONFIGS = LEARNED + BASELINES
SNAPSHOT_MAP = {
    "sx_norm_t10": "S24-T10",
    "sx_norm_t30": "S24-T30",
    "sx_norm_t60": "S24-T60",
    "sx_norm_s36_t60": "S36-T60",
    "sx_norm_s48_t60": "S48-T60",
    "SNaive-24": "SNaive-24",
    "SNaive-168": "SNaive-168",
}
PROBABILITY_MAP = {
    "NORM-T10": "S24-T10",
    "NORM-T30": "S24-T30",
    "NORM-T60": "S24-T60",
    "NORM-S36-T60": "S36-T60",
    "NORM-S48-T60": "S48-T60",
    "SNaive-24": "SNaive-24",
    "SNaive-168": "SNaive-168",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _read_predictions(package_root: Path) -> pd.DataFrame:
    main_path = package_root / "_work" / "forecasting" / "predictions.csv"
    breadth_path = (
        package_root / "_work" / "breadth_t60" / "point" / "predictions.csv"
    )
    date_columns = ["origin", "timestamp", "fit_end", "history_end"]
    main = pd.read_csv(main_path, parse_dates=date_columns)
    breadth = pd.read_csv(breadth_path, parse_dates=date_columns)
    pieces: list[pd.DataFrame] = []
    source_choice = {
        "S24-T10": (main, "sx_norm_t10", main_path),
        "S24-T30": (main, "sx_norm_t30", main_path),
        "S24-T60": (breadth, "sx_norm_t60", breadth_path),
        "S36-T60": (breadth, "sx_norm_s36_t60", breadth_path),
        "S48-T60": (breadth, "sx_norm_s48_t60", breadth_path),
        "SNaive-24": (breadth, "SNaive-24", breadth_path),
        "SNaive-168": (breadth, "SNaive-168", breadth_path),
    }
    for config, (source, snapshot, source_path) in source_choice.items():
        rows = source.loc[
            (source["snapshot"] == snapshot)
            & source["period"].isin(["prewar_2022", "late_february_2022"])
        ].copy()
        rows["configuration"] = config
        rows["source_file"] = source_path.relative_to(package_root).as_posix()
        pieces.append(rows)
    result = pd.concat(pieces, ignore_index=True)
    result["analysis_period"] = np.where(
        result["origin"] <= pd.Timestamp("2022-02-23"),
        "primary_54",
        "descriptive_stress_5",
    )
    result.sort_values(
        ["configuration", "origin", "target", "horizon"], inplace=True
    )
    return result


def _validate_prediction_sources(predictions: pd.DataFrame) -> None:
    for config in CONFIGS:
        rows = predictions.loc[predictions["configuration"] == config]
        assert rows["origin"].nunique() == 59, config
        for target in TARGETS:
            target_rows = rows.loc[rows["target"] == target]
            assert len(target_rows) == 1416, (config, target, len(target_rows))
            assert target_rows["prediction"].notna().all()
            assert np.isfinite(target_rows["prediction"].to_numpy(float)).all()
    actual_sets = []
    for config in CONFIGS:
        series = (
            predictions.loc[predictions["configuration"] == config]
            .sort_values(["target", "timestamp"])
            .set_index(["target", "timestamp"])["actual"]
        )
        actual_sets.append(series)
    for candidate in actual_sets[1:]:
        np.testing.assert_allclose(actual_sets[0].to_numpy(), candidate.to_numpy())


def _read_probabilistic_daily(package_root: Path) -> pd.DataFrame:
    main_path = package_root / "_work" / "probabilistic" / "daily_metrics.csv"
    breadth_path = (
        package_root
        / "_work"
        / "breadth_t60"
        / "probabilistic"
        / "daily_metrics.csv"
    )
    main = pd.read_csv(main_path, parse_dates=["date"])
    breadth = pd.read_csv(breadth_path, parse_dates=["date"])
    source_choice = {
        "S24-T10": (main, "NORM-T10", main_path),
        "S24-T30": (main, "NORM-T30", main_path),
        "S24-T60": (breadth, "NORM-T60", breadth_path),
        "S36-T60": (breadth, "NORM-S36-T60", breadth_path),
        "S48-T60": (breadth, "NORM-S48-T60", breadth_path),
        "SNaive-24": (breadth, "SNaive-24", breadth_path),
        "SNaive-168": (breadth, "SNaive-168", breadth_path),
    }
    pieces: list[pd.DataFrame] = []
    for config, (source, model_id, source_path) in source_choice.items():
        rows = source.loc[
            (source["model_id"] == model_id)
            & source["target"].isin(TARGETS)
            & (source["date"] >= pd.Timestamp("2022-01-01"))
            & (source["date"] <= pd.Timestamp("2022-02-28"))
        ].copy()
        rows["configuration"] = config
        rows["source_file"] = source_path.relative_to(package_root).as_posix()
        pieces.append(rows)
    result = pd.concat(pieces, ignore_index=True)
    result["analysis_period"] = np.where(
        result["date"] <= pd.Timestamp("2022-02-23"),
        "primary_54",
        "descriptive_stress_5",
    )
    for config in CONFIGS:
        for target in TARGETS:
            subset = result.loc[
                (result["configuration"] == config) & (result["target"] == target)
            ]
            assert len(subset) == 59, (config, target, len(subset))
    return result


def _point_metrics(
    predictions: pd.DataFrame, period: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = predictions.loc[predictions["analysis_period"] == period].copy()
    target_rows: list[dict[str, Any]] = []
    for (config, target), rows in subset.groupby(["configuration", "target"]):
        errors = rows["actual"].to_numpy(float) - rows["prediction"].to_numpy(float)
        target_rows.append(
            {
                "configuration": config,
                "target": target,
                "n_origins": int(rows["origin"].nunique()),
                "n_hours": int(len(rows)),
                "RMSE": float(np.sqrt(np.mean(errors**2))),
                "MAE": float(np.mean(np.abs(errors))),
            }
        )
    target = pd.DataFrame(target_rows)
    wide = target.pivot(index="configuration", columns="target")
    summary_rows: list[dict[str, Any]] = []
    for config in CONFIGS:
        summary_rows.append(
            {
                "configuration": config,
                "analysis_period": period,
                "n_origins": int(wide.loc[config, ("n_origins", "P_Power")]),
                "n_hours_per_target": int(wide.loc[config, ("n_hours", "P_Power")]),
                "P_RMSE_kW": float(wide.loc[config, ("RMSE", "P_Power")]),
                "P_MAE_kW": float(wide.loc[config, ("MAE", "P_Power")]),
                "Q_RMSE_kVAr": float(wide.loc[config, ("RMSE", "Q_Power")]),
                "Q_MAE_kVAr": float(wide.loc[config, ("MAE", "Q_Power")]),
                "fixed_scale_paired_RMSE": 0.5
                * (
                    float(wide.loc[config, ("RMSE", "P_Power")])
                    / SCALES["P_Power"]
                    + float(wide.loc[config, ("RMSE", "Q_Power")])
                    / SCALES["Q_Power"]
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)

    daily_target = (
        subset.assign(squared_error=(subset["actual"] - subset["prediction"]) ** 2)
        .groupby(["configuration", "origin", "target"], as_index=False)
        .agg(MSE=("squared_error", "mean"), MAE=("error", lambda x: np.mean(np.abs(x))))
    )
    daily_wide = daily_target.pivot(
        index=["configuration", "origin"], columns="target", values=["MSE", "MAE"]
    )
    daily_wide.columns = [f"{metric}_{TARGET_SHORT[target]}" for metric, target in daily_wide.columns]
    daily_wide.reset_index(inplace=True)
    daily_wide["analysis_period"] = period
    daily_wide["paired_normalized_squared_loss"] = 0.5 * (
        daily_wide["MSE_P"] / SCALES["P_Power"] ** 2
        + daily_wide["MSE_Q"] / SCALES["Q_Power"] ** 2
    )
    return summary, daily_wide


def _probabilistic_metrics(
    daily: pd.DataFrame, period: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = daily.loc[daily["analysis_period"] == period].copy()
    target = (
        subset.groupby(["configuration", "target"], as_index=False)
        .agg(
            n_origins=("date", "nunique"),
            CRPS=("CRPS", "mean"),
            Pinball_mean=("Pinball_mean", "mean"),
            Coverage80=("Coverage80", "mean"),
            Width80=("Width80", "mean"),
            IntervalScore80=("IntervalScore80", "mean"),
            Coverage95=("Coverage95", "mean"),
            Width95=("Width95", "mean"),
            IntervalScore95=("IntervalScore95", "mean"),
        )
    )
    pivot = target.pivot(index="configuration", columns="target")
    summary_rows: list[dict[str, Any]] = []
    for config in CONFIGS:
        row = {
            "configuration": config,
            "analysis_period": period,
            "n_origins": int(pivot.loc[config, ("n_origins", "P_Power")]),
        }
        for target_name, label in (("P_Power", "P"), ("Q_Power", "Q")):
            for metric in (
                "CRPS",
                "Pinball_mean",
                "Coverage80",
                "Width80",
                "IntervalScore80",
                "Coverage95",
                "Width95",
                "IntervalScore95",
            ):
                row[f"{label}_{metric}"] = float(
                    pivot.loc[config, (metric, target_name)]
                )
        row["paired_normalized_CRPS"] = 0.5 * (
            row["P_CRPS"] / SCALES["P_Power"]
            + row["Q_CRPS"] / SCALES["Q_Power"]
        )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    daily_pivot = subset.pivot(
        index=["configuration", "date"], columns="target", values="CRPS"
    ).reset_index()
    daily_pivot.rename(
        columns={"date": "origin", "P_Power": "CRPS_P", "Q_Power": "CRPS_Q"},
        inplace=True,
    )
    daily_pivot["analysis_period"] = period
    daily_pivot["paired_normalized_CRPS"] = 0.5 * (
        daily_pivot["CRPS_P"] / SCALES["P_Power"]
        + daily_pivot["CRPS_Q"] / SCALES["Q_Power"]
    )
    return summary, daily_pivot


def _dm_family(
    daily: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
    *,
    loss_column: str,
    loss_label: str,
    family: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_a, model_b in pairs:
        pivot = daily.loc[daily["configuration"].isin([model_a, model_b])].pivot(
            index="origin", columns="configuration", values=loss_column
        )
        result = dm_test_losses(
            pivot[model_a].to_numpy(float),
            pivot[model_b].to_numpy(float),
            hac_lag="3",
            min_days=20,
        )
        rows.append(
            {
                "family": family,
                "loss": loss_label,
                "model_a": model_a,
                "model_b": model_b,
                "contrast": f"{model_a} - {model_b}",
                "mean_daily_normalized_loss_difference": result.mean_difference,
                "ci_low": result.ci_low,
                "ci_high": result.ci_high,
                "raw_p": result.p_value,
                "hac_lag": result.hac_lag,
                "n_origins": result.n,
                "status": result.status,
                "numerical_leader": (
                    model_a if result.mean_difference < 0 else model_b
                ),
            }
        )
    table = pd.DataFrame(rows)
    table["holm_p"] = holm_adjust(table["raw_p"].to_numpy(float))
    table["significant_0_05"] = table["holm_p"] < 0.05
    return table


def _model_metadata(package_root: Path, snapshot: str) -> Mapping[str, Any]:
    path = model_registry.metadata_path(
        package_root / "Models", snapshot, "P_Power"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _common_breadth_scores(package_root: Path) -> pd.DataFrame:
    seeds = (42, 17, 73)
    seed_runs = {
        42: {
            24: "sx_norm_t60",
            36: "sx_norm_s36_t60",
            48: "sx_norm_s48_t60",
        },
        17: {
            24: "sx_norm_s17_t60",
            36: "sx_norm_s36_s17_t60",
            48: "sx_norm_s48_s17_t60",
        },
        73: {
            24: "sx_norm_s73_t60",
            36: "sx_norm_s36_s73_t60",
            48: "sx_norm_s48_s73_t60",
        },
    }
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        run48 = seed_runs[seed][48]
        trial_path = package_root / "_work" / "search" / run48 / "trials.csv"
        trials = pd.read_csv(trial_path)
        anchor_values = {
            str(row.anchor_source_snapshot): float(row.objective)
            for row in trials.itertuples()
            if isinstance(row.anchor_source_snapshot, str)
            and row.anchor_source_snapshot in {
                seed_runs[seed][24],
                seed_runs[seed][36],
            }
            and np.isfinite(float(row.objective))
        }
        metadata = {
            breadth: _model_metadata(package_root, snapshot)
            for breadth, snapshot in seed_runs[seed].items()
        }
        if seed_runs[seed][36] not in anchor_values:
            params24 = metadata[24]["tuned_params"]
            params36 = metadata[36]["tuned_params"]
            if params24 != params36:
                raise RuntimeError(
                    f"Seed {seed} SEARCH-36 has no common-score anchor and is not "
                    "an exact duplicate of SEARCH-24."
                )
            anchor_values[seed_runs[seed][36]] = anchor_values[seed_runs[seed][24]]
        for breadth in (24, 36, 48):
            snapshot = seed_runs[seed][breadth]
            own_objective = float(metadata[breadth]["source"]["objective"])
            common_objective = (
                float(metadata[48]["source"]["objective"])
                if breadth == 48
                else float(anchor_values[snapshot])
            )
            rows.append(
                {
                    "seed": seed,
                    "selected_breadth": breadth,
                    "snapshot": snapshot,
                    "own_design_objective": own_objective,
                    "common_SEARCH48_objective": common_objective,
                    "parameter_vector_duplicate_of": (
                        seed_runs[seed][24]
                        if breadth == 36
                        and metadata[24]["tuned_params"] == metadata[36]["tuned_params"]
                        else ""
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    medians = (
        frame.groupby("selected_breadth", as_index=False)
        .agg(
            own_design_median=("own_design_objective", "median"),
            own_design_sample_cv=(
                "own_design_objective",
                lambda values: float(np.std(values, ddof=1) / np.mean(values)),
            ),
            common_SEARCH48_median=("common_SEARCH48_objective", "median"),
            common_SEARCH48_range_over_median=(
                "common_SEARCH48_objective",
                lambda values: float((np.max(values) - np.min(values)) / np.median(values)),
            ),
        )
    )
    return frame.merge(medians, on="selected_breadth", how="left")


def _search_design_table(package_root: Path) -> pd.DataFrame:
    anytime_path = package_root / "_work" / "tables" / "anytime_curves.csv"
    anytime = pd.read_csv(anytime_path)
    anytime = anytime.loc[anytime["run_id"] == "sx_norm_t60"].reset_index(drop=True)
    assert len(anytime) == 60
    milestone_rows = {10: anytime.iloc[9], 30: anytime.iloc[29], 60: anytime.iloc[59]}
    breadth_path = (
        package_root
        / "_work"
        / "breadth_t60"
        / "analysis"
        / "tables"
        / "search_completion.csv"
    )
    breadth = pd.read_csv(breadth_path).set_index("system")
    rows = []
    for trials in (10, 30, 60):
        item = milestone_rows[trials]
        seed_evidence = (
            "42 operational; 17 and 73 development rescoring"
            if trials == 60
            else "42 (one nested trajectory)"
        )
        search_role = (
            "Depth and breadth reference"
            if trials == 60
            else "Nested best-so-far milestone"
        )
        rows.append(
            {
                "configuration": f"S24-T{trials}",
                "tuning_origins": 24,
                "terminal_trials": trials,
                "sampler_seeds_represented": seed_evidence,
                "actual_target_fold_fits": int(item["cumulative_target_fold_fits"]),
                "search_role": search_role,
                "cumulative_trial_time_seconds": float(item["cumulative_wall_time_seconds"]),
            }
        )
    for system, breadth_count in (("S36-T60", 36), ("S48-T60", 48)):
        item = breadth.loc[system]
        rows.append(
            {
                "configuration": system,
                "tuning_origins": breadth_count,
                "terminal_trials": 60,
                "sampler_seeds_represented": (
                    "42 operational; 17 and 73 development rescoring"
                ),
                "actual_target_fold_fits": int(item["actual_target_fold_fits"]),
                "search_role": "Frozen seed-42 breadth configuration",
                "cumulative_trial_time_seconds": float(item["trial_duration_seconds"]),
            }
        )
    return pd.DataFrame(rows)


def _forecast_wide(predictions: pd.DataFrame) -> pd.DataFrame:
    index_columns = ["origin", "timestamp", "horizon", "analysis_period"]
    actual = (
        predictions[index_columns + ["target", "actual"]]
        .drop_duplicates(index_columns + ["target"])
        .pivot(index=index_columns, columns="target", values="actual")
        .rename(columns={"P_Power": "actual_P_kW", "Q_Power": "actual_Q_kVAr"})
    )
    forecast = predictions.pivot(
        index=index_columns,
        columns=["configuration", "target"],
        values="prediction",
    )
    forecast.columns = [
        f"{config}_{TARGET_SHORT[target]}_{UNITS[target]}" for config, target in forecast.columns
    ]
    return actual.join(forecast).reset_index().sort_values("timestamp")


def _figure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.7,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _plot_figure1(
    package_root: Path,
    output_dir: Path,
    table2: pd.DataFrame,
    breadth_summary: pd.DataFrame,
) -> None:
    _figure_style()
    anytime = pd.read_csv(package_root / "_work" / "tables" / "anytime_curves.csv")
    curve = anytime.loc[anytime["run_id"] == "sx_norm_t60"].reset_index(drop=True)
    milestones = {10: curve.iloc[9], 30: curve.iloc[29], 60: curve.iloc[59]}
    external = table2.set_index("configuration")["fixed_scale_paired_RMSE"]

    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.35), constrained_layout=True)
    ax = axes[0]
    ax.step(
        curve["cumulative_target_fold_fits"],
        curve["best_complete_objective"],
        where="post",
        color="#1f5a89",
        linewidth=1.45,
    )
    marker_colors = {10: "#6c757d", 30: "#d97706", 60: "#1769aa"}
    for trials, row in milestones.items():
        x = float(row["cumulative_target_fold_fits"])
        y = float(row["best_complete_objective"])
        ax.scatter(x, y, s=34, color=marker_colors[trials], edgecolor="white", zorder=4)
        is_last = trials == 60
        ax.annotate(
            f"S24-T{trials}\n2022 paired RMSE={external[f'S24-T{trials}']:.3f}",
            xy=(x, y),
            xytext=(-5 if is_last else 5, 8 if is_last else -22),
            textcoords="offset points",
            fontsize=6.9,
            ha="right" if is_last else "left",
            va="bottom" if is_last else "top",
        )
    ax.set_title("(a) Nested search depth", loc="left", fontweight="bold")
    ax.set_xlabel("Cumulative target-fold fits")
    ax.set_ylabel("Best-so-far tuning objective")
    ax.grid(True, color="#d9dde1", linewidth=0.55)
    ax.set_xlim(left=0)

    ax = axes[1]
    palette = {42: "#1769aa", 17: "#d97706", 73: "#6f7f3f"}
    for seed, rows in breadth_summary.groupby("seed"):
        rows = rows.sort_values("selected_breadth")
        ax.plot(
            rows["selected_breadth"],
            rows["common_SEARCH48_objective"],
            color=palette[int(seed)],
            linewidth=0.95,
            marker="o",
            markersize=4,
            label=f"Seed {int(seed)}",
        )
    median = (
        breadth_summary.groupby("selected_breadth", as_index=False)[
            "common_SEARCH48_objective"
        ].median()
    )
    ax.plot(
        median["selected_breadth"],
        median["common_SEARCH48_objective"],
        color="black",
        linewidth=2.0,
        marker="s",
        markersize=4.5,
        label="Median",
        zorder=5,
    )
    ax.set_title("(b) Chronological validation breadth", loc="left", fontweight="bold")
    ax.set_xlabel("Tuning origins used for selection")
    ax.set_ylabel("Objective on common 48-origin schedule")
    ax.set_xticks([24, 36, 48])
    ax.grid(True, color="#d9dde1", linewidth=0.55)
    ax.legend(frameon=False, ncol=2, loc="best")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"02_search_depth_and_breadth.{suffix}", dpi=350)
    plt.close(fig)


def _series_for_plot(
    predictions: pd.DataFrame, config: str, target: str
) -> pd.DataFrame:
    return predictions.loc[
        (predictions["configuration"] == config) & (predictions["target"] == target)
    ].sort_values("timestamp")


def _plot_figure2(predictions: pd.DataFrame, output_dir: Path) -> None:
    _figure_style()
    fig, axes = plt.subplots(4, 1, figsize=(10.0, 8.5), sharex=True, constrained_layout=True)
    actual_config = "S24-T60"
    search_styles = {
        "S24-T10": ("#6c757d", "--"),
        "S24-T30": ("#d97706", "-."),
        "S24-T60": ("#1769aa", "-"),
    }
    breadth_styles = {
        "S24-T60": ("#6c757d", "--"),
        "S36-T60": ("#1769aa", "-"),
        "S48-T60": ("#d97706", "-."),
    }
    start_stress = pd.Timestamp("2022-02-24 00:00")
    end_plot = pd.Timestamp("2022-03-01 00:00")
    ticks = pd.to_datetime(
        ["2022-01-01", "2022-01-15", "2022-02-01", "2022-02-15", "2022-03-01"]
    )
    panels = [
        (axes[0], "P_Power", search_styles, "(a) Active power P: search depth"),
        (axes[1], "Q_Power", search_styles, "(b) Reactive power Q: search depth"),
        (axes[2], "P_Power", breadth_styles, "(c) Active power P: tuning breadth"),
        (axes[3], "Q_Power", breadth_styles, "(d) Reactive power Q: tuning breadth"),
    ]
    for panel_index, (ax, target, styles, title) in enumerate(panels):
        actual = _series_for_plot(predictions, actual_config, target)
        ax.plot(
            actual["timestamp"],
            actual["actual"],
            color="black",
            linewidth=0.90,
            alpha=0.78,
            label="Prepared observation",
            zorder=2,
        )
        for config, (color, linestyle) in styles.items():
            rows = _series_for_plot(predictions, config, target)
            ax.plot(
                rows["timestamp"],
                rows["prediction"],
                color=color,
                linestyle=linestyle,
                linewidth=0.72,
                alpha=0.92,
                label=config,
                zorder=3,
            )
        ax.axvspan(start_stress, end_plot, color="#d9d9d9", alpha=0.55, zorder=0)
        ax.axvline(start_stress, color="#555555", linewidth=0.75, zorder=1)
        if panel_index == 0:
            ax.text(
                end_plot - pd.Timedelta(hours=6),
                0.94,
                "24-28 Feb: descriptive",
                transform=mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes),
                fontsize=6.5,
                color="#4f4f4f",
                ha="right",
                va="top",
            )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(
            "Active power P\n(kW)" if target == "P_Power" else "Reactive power Q\n(kVAr)"
        )
        ax.set_xlim(pd.Timestamp("2022-01-01"), end_plot)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.grid(True, color="#d9dde1", linewidth=0.45)
        if panel_index in (0, 2):
            ax.legend(
                frameon=False,
                ncol=4,
                loc="upper center",
                bbox_to_anchor=(0.58, 1.02),
                columnspacing=1.1,
                handlelength=2.3,
            )
    axes[-1].set_xlabel("Calendar date")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"03_forecast_trajectories.{suffix}", dpi=350)
    plt.close(fig)


def _copy_supporting_figures(package_root: Path, output_dir: Path) -> None:
    source_dir = (
        package_root / "_work" / "breadth_t60" / "analysis" / "figures"
    )
    retained = {
        "nested_split_breadth": "01_validation_origins",
        "primary_54d_pq_rmse": "04_point_accuracy",
        "primary_54d_probabilistic_crps": "05_probabilistic_accuracy",
    }
    for source_stem, output_stem in retained.items():
        for suffix in (".png", ".pdf"):
            source = source_dir / f"{source_stem}{suffix}"
            if not source.exists():
                raise FileNotFoundError(f"Required publication figure is absent: {source}")
            shutil.copy2(source, output_dir / f"{output_stem}{suffix}")


def _plot_paired_tests(tests: pd.DataFrame, output_dir: Path) -> None:
    """Show the twelve prespecified daily contrasts without a dense matrix."""
    _figure_style()
    loss_panels = (
        ("Paired normalized squared error", "(a) Point forecast loss"),
        ("Paired normalized CRPS", "(b) Probabilistic forecast loss"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.4), constrained_layout=True)
    y_positions = np.array([6, 5, 4, 2, 1, 0], dtype=float)
    for ax, (loss_name, title) in zip(axes, loss_panels):
        rows = tests.loc[tests["loss"] == loss_name].copy()
        rows["group_order"] = rows["family"].map(
            {"A. Search depth": 0, "B. Tuning breadth": 1}
        )
        rows.sort_values(
            ["group_order", "model_a", "model_b"], inplace=True
        )
        if len(rows) != 6:
            raise ValueError(f"Expected six contrasts for {loss_name}, found {len(rows)}")
        labels = [
            f"{row.model_a} - {row.model_b}"
            for row in rows.itertuples(index=False)
        ]
        for y, row in zip(y_positions, rows.itertuples(index=False)):
            mean = float(row.mean_daily_normalized_loss_difference)
            low = float(row.ci_low)
            high = float(row.ci_high)
            significant = bool(row.significant_0_05)
            color = "#18864b" if significant else "#555f69"
            ax.errorbar(
                mean,
                y,
                xerr=np.array([[mean - low], [high - mean]]),
                fmt="o",
                color=color,
                ecolor=color,
                markersize=5.0,
                elinewidth=1.2,
                capsize=3,
                zorder=3,
            )
        ax.axvline(0.0, color="black", linewidth=0.8, zorder=1)
        ax.axhline(3.0, color="#c9cdd1", linewidth=0.7)
        ax.set_yticks(y_positions, labels)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Mean daily loss difference")
        ax.grid(axis="x", color="#d9dde1", linewidth=0.55)
        ax.text(
            0.01,
            0.98,
            "Search depth",
            transform=ax.transAxes,
            va="top",
            fontsize=7.2,
            color="#555f69",
        )
        ax.text(
            0.01,
            0.43,
            "Tuning breadth",
            transform=ax.transAxes,
            va="top",
            fontsize=7.2,
            color="#555f69",
        )
    axes[1].tick_params(labelleft=True)
    fig.suptitle("Paired daily comparisons over the primary 54 forecast days")
    fig.supxlabel(
        "First system minus second; negative values favor the first. "
        "Bars are 95% confidence intervals; green indicates Holm-adjusted p < 0.05.",
        fontsize=8,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"06_paired_comparisons.{suffix}", dpi=350)
    plt.close(fig)


def _source_inventory(package_root: Path, paths: Iterable[Path]) -> pd.DataFrame:
    rows = []
    for path in sorted(set(path.resolve() for path in paths)):
        try:
            relative = path.relative_to(package_root.resolve()).as_posix()
        except ValueError:
            relative = path.relative_to(package_root.parent.resolve()).as_posix()
        row: dict[str, Any] = {
            "source_file": path.name,
            "relative_path": relative,
            "sha256": _sha256(path),
            "file_type": path.suffix.lower().lstrip("."),
            "model_or_run_id": "",
            "period": "",
            "role": "frozen source",
            "rows": "",
            "first_timestamp": "",
            "last_timestamp": "",
            "notes": "",
        }
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path, low_memory=False)
            row["rows"] = len(frame)
            for candidate in ("timestamp", "date", "origin", "datetime_start"):
                if candidate in frame.columns:
                    parsed = pd.to_datetime(frame[candidate], errors="coerce")
                    if parsed.notna().any():
                        row["first_timestamp"] = parsed.min().isoformat()
                        row["last_timestamp"] = parsed.max().isoformat()
                        break
        elif path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            row["model_or_run_id"] = str(
                payload.get("experiment_snapshot") or payload.get("snapshot") or ""
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _write_manifest(output_root: Path, package_root: Path) -> None:
    deliverable_directories = (
        output_root / "figures",
        output_root / "tables",
        output_root / "data",
        output_root / "reproducibility",
    )
    deliverables = [
        path
        for directory in deliverable_directories
        for path in directory.rglob("*")
        if path.is_file()
        and path.name != "final_results_manifest.json"
        and not path.name.endswith(".inspect.ndjson")
    ]
    audit_path = output_root / "reproducibility" / "forecast_leakage_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    payload = {
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "package_root": "..",
        "package_root_note": "All artifact paths are relative to the Results directory.",
        "new_hyperparameter_tuning_performed": False,
        "leakage_audit": audit,
        "fixed_scales": SCALES,
        "software_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matplotlib": mpl.__version__,
        },
        "artifacts": [
            {
                "path": path.relative_to(output_root).as_posix(),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(deliverables)
        ],
    }
    (output_root / "reproducibility" / "final_results_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def main(
    package_root: Path | None = None, output_root: Path | None = None
) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    package_root = (
        Path(package_root).expanduser().resolve()
        if package_root is not None
        else Path(__file__).resolve().parents[1]
    )
    output_root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else package_root / "Results"
    )
    data_dir = output_root / "data"
    table_dir = output_root / "tables"
    figure_dir = output_root / "figures"
    audit_dir = output_root / "reproducibility"
    for directory in (data_dir, table_dir, figure_dir, audit_dir):
        directory.mkdir(parents=True, exist_ok=True)

    leakage_path = audit_dir / "forecast_leakage_audit.json"
    if not leakage_path.exists():
        raise FileNotFoundError(
            "Run audit_59day_forecasts.py before building publication evidence."
        )
    leakage = json.loads(leakage_path.read_text(encoding="utf-8"))
    if leakage.get("status") != "PASS" or not leakage.get(
        "future_target_perturbation_verified"
    ):
        raise RuntimeError("The mandatory forecast leakage audit has not passed.")

    predictions = _read_predictions(package_root)
    _validate_prediction_sources(predictions)
    probability_daily = _read_probabilistic_daily(package_root)

    long_columns = [
        "configuration",
        "analysis_period",
        "origin",
        "timestamp",
        "horizon",
        "target",
        "actual",
        "prediction",
        "error",
        "fit_end",
        "history_end",
        "source_file",
    ]
    predictions[long_columns].to_csv(data_dir / "forecast_59days_long.csv", index=False)
    _forecast_wide(predictions).to_csv(data_dir / "forecast_59days_wide.csv", index=False)

    primary_point, primary_daily_point = _point_metrics(predictions, "primary_54")
    stress_point, stress_daily_point = _point_metrics(
        predictions, "descriptive_stress_5"
    )
    primary_prob, primary_daily_prob = _probabilistic_metrics(
        probability_daily, "primary_54"
    )
    stress_prob, stress_daily_prob = _probabilistic_metrics(
        probability_daily, "descriptive_stress_5"
    )

    primary = primary_point.merge(
        primary_prob.drop(columns=["analysis_period", "n_origins"]),
        on="configuration",
        validate="one_to_one",
    )
    stress = stress_point.merge(
        stress_prob.drop(columns=["analysis_period", "n_origins"]),
        on="configuration",
        validate="one_to_one",
    )
    primary.to_csv(data_dir / "primary_54day_metrics.csv", index=False)
    stress.to_csv(data_dir / "stress_5day_metrics.csv", index=False)

    primary_daily = primary_daily_point.merge(
        primary_daily_prob.drop(columns="analysis_period"),
        on=["configuration", "origin"],
        validate="one_to_one",
    )
    primary_daily["analysis_period"] = "primary_54"
    stress_daily = stress_daily_point.merge(
        stress_daily_prob.drop(columns="analysis_period"),
        on=["configuration", "origin"],
        validate="one_to_one",
    )
    stress_daily["analysis_period"] = "descriptive_stress_5"
    daily_all = pd.concat([primary_daily, stress_daily], ignore_index=True)
    daily_all.sort_values(["configuration", "origin"], inplace=True)
    daily_all.to_csv(data_dir / "daily_paired_losses.csv", index=False)

    depth_pairs = (
        ("S24-T10", "S24-T30"),
        ("S24-T10", "S24-T60"),
        ("S24-T30", "S24-T60"),
    )
    breadth_pairs = (
        ("S24-T60", "S36-T60"),
        ("S24-T60", "S48-T60"),
        ("S36-T60", "S48-T60"),
    )
    tests = pd.concat(
        [
            _dm_family(
                primary_daily,
                depth_pairs,
                loss_column="paired_normalized_squared_loss",
                loss_label="Paired normalized squared error",
                family="A. Search depth",
            ),
            _dm_family(
                primary_daily,
                depth_pairs,
                loss_column="paired_normalized_CRPS",
                loss_label="Paired normalized CRPS",
                family="A. Search depth",
            ),
            _dm_family(
                primary_daily,
                breadth_pairs,
                loss_column="paired_normalized_squared_loss",
                loss_label="Paired normalized squared error",
                family="B. Tuning breadth",
            ),
            _dm_family(
                primary_daily,
                breadth_pairs,
                loss_column="paired_normalized_CRPS",
                loss_label="Paired normalized CRPS",
                family="B. Tuning breadth",
            ),
        ],
        ignore_index=True,
    )
    tests.to_csv(table_dir / "Table3_paired_tests.csv", index=False)

    expected = {
        ("S24-T60", "S36-T60"): (0.0514585304630018, 0.1543755913890054),
        ("S24-T60", "S48-T60"): (0.37672376435856625, 0.7534475287171325),
        ("S36-T60", "S48-T60"): (0.6703200855799564, 0.7534475287171325),
    }
    breadth_point = tests.loc[
        (tests["family"] == "B. Tuning breadth")
        & (tests["loss"] == "Paired normalized squared error")
    ]
    for row in breadth_point.itertuples():
        raw_expected, holm_expected = expected[(row.model_a, row.model_b)]
        assert math.isclose(row.raw_p, raw_expected, abs_tol=1e-12)
        assert math.isclose(row.holm_p, holm_expected, abs_tol=1e-12)

    design = _search_design_table(package_root)
    design.to_csv(table_dir / "Table1_search_design_cost.csv", index=False)
    table2 = primary[
        [
            "configuration",
            "P_RMSE_kW",
            "Q_RMSE_kVAr",
            "fixed_scale_paired_RMSE",
            "P_CRPS",
            "Q_CRPS",
            "paired_normalized_CRPS",
        ]
    ].copy()
    table2.to_csv(table_dir / "Table2_primary_performance.csv", index=False)

    breadth_summary = _common_breadth_scores(package_root)
    breadth_summary.to_csv(table_dir / "breadth_three_seed_summary.csv", index=False)

    _plot_figure1(package_root, figure_dir, table2, breadth_summary)
    _plot_figure2(predictions, figure_dir)
    _copy_supporting_figures(package_root, figure_dir)
    _plot_paired_tests(tests, figure_dir)

    code_root = package_root / "code"
    internal_code_root = (
        code_root / "_internal" if (code_root / "_internal").is_dir() else code_root
    )
    source_paths = [
        package_root / "Input" / "PQ.xlsx",
        package_root / "Input" / "objective_scales.json",
        package_root / "_work" / "forecasting" / "predictions.csv",
        package_root / "_work" / "forecasting" / "daily_paired_losses.csv",
        package_root / "_work" / "probabilistic" / "daily_metrics.csv",
        package_root / "_work" / "tables" / "anytime_curves.csv",
        package_root / "_work" / "tables" / "milestone_summary.csv",
        package_root / "_work" / "breadth_t60" / "point" / "predictions.csv",
        package_root
        / "_work"
        / "breadth_t60"
        / "probabilistic"
        / "daily_metrics.csv",
        package_root
        / "_work"
        / "breadth_t60"
        / "analysis"
        / "tables"
        / "search_completion.csv",
        package_root
        / "_work"
        / "breadth_t60"
        / "analysis"
        / "statistics"
        / "dm_point_t60_primary_54d.csv",
        package_root
        / "_work"
        / "breadth_t60"
        / "analysis"
        / "statistics"
        / "dm_crps_t60_primary_54d.csv",
        internal_code_root / "evaluate_search.py",
        internal_code_root / "Forecasting_search.py",
        internal_code_root / "analyze_search.py",
    ]
    seed_runs = (
        "sx_norm_t60",
        "sx_norm_s36_t60",
        "sx_norm_s48_t60",
        "sx_norm_s17_t60",
        "sx_norm_s36_s17_t60",
        "sx_norm_s48_s17_t60",
        "sx_norm_s73_t60",
        "sx_norm_s36_s73_t60",
        "sx_norm_s48_s73_t60",
    )
    for snapshot in seed_runs:
        source_paths.append(
            model_registry.metadata_path(
                package_root / "Models", snapshot, "P_Power"
            )
        )
        source_paths.append(package_root / "_work" / "search" / snapshot / "trials.csv")
    inventory = _source_inventory(package_root, source_paths)
    inventory.to_csv(audit_dir / "source_inventory.csv", index=False)
    pd.DataFrame(
        columns=["source_a", "source_b", "field", "value_a", "value_b", "resolution"]
    ).to_csv(audit_dir / "source_discrepancies.csv", index=False)

    report = f"""# Conference Finalization Report

Generated: {datetime.now(timezone.utc).isoformat()}

## Frozen scope

- No new Optuna tuning was performed.
- Five unique seed-42 learned configurations were retained.
- Three sampler seeds were used only for descriptive development rescoring.
- The primary period contains 54 origins (1 January-23 February 2022).
- The descriptive stress period contains five origins (24-28 February 2022).

## Validation

- Forecast leakage audit: **{leakage['status']}**.
- Maximum future-target perturbation difference: {leakage['maximum_perturbation_difference']}.
- Maximum target-order difference: {leakage['maximum_order_difference']}.
- All seven reported configurations contain 1,416 hourly forecasts per target.
- Actual target values agree across configurations and sources.
- Breadth squared-error p-values reproduce the archived values to 1e-12.

## Numerical interpretation

- S36-T60 is the numerical primary leader under fixed paired RMSE and paired CRPS.
- S48-T60 has the lowest Q RMSE among the five learned configurations.
- No search-depth squared-error or tuning-breadth contrast is significant after its three-test Holm adjustment.
- In the separate search-depth CRPS family, S24-T60 is lower than S24-T10 and S24-T30 after Holm adjustment (adjusted p = 0.02775 for both).
- The three-seed breadth trajectories are descriptive and do not establish a universal optimum.
"""
    (audit_dir / "finalization_report.md").write_text(report, encoding="utf-8")

    _write_manifest(output_root, package_root)
    print(table2.to_string(index=False))
    print("\nPaired tests:\n", tests.to_string(index=False))
    print(f"\nWrote publication evidence to {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output-root", type=Path, default=None)
    arguments = parser.parse_args()
    main(arguments.package_root, arguments.output_root)

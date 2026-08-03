# -*- coding: utf-8 -*-
"""Statistical analysis and figures for the LightGBM search experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import t as student_t

import evaluate_search as evaluator
import model_registry


SCRIPT_VERSION = "search-experiment-1.1"
PRIMARY_PERIOD = "prewar_2022"
PRIMARY_LOSS = "paired_normalized_squared_loss"
FULL_2022_PERIOD = "jan_feb_2022"
FULL_2022_COMPONENTS = ("prewar_2022", "late_february_2022")
PLANNED_CONTRASTS = (
    ("sx_norm_t60", "sx_raw_t60", "normalized TPE 60 vs raw TPE 60"),
    ("sx_norm_t30", "sx_random_t30", "normalized TPE 30 vs random 30"),
    ("sx_norm_t60", "sx_gap48_t30", "normalized TPE 60 vs gap-robust TPE 30"),
    ("sx_norm_t60", "sx_alt_c3", "normalized shared TPE 60 vs alternating cycle 3"),
    ("sx_norm_t10", "sx_norm_t30", "normalized TPE 10 vs 30"),
    ("sx_norm_t30", "sx_norm_t60", "normalized TPE 30 vs 60"),
    ("sx_alt_c1", "sx_alt_c2", "alternating cycle 1 vs 2"),
    ("sx_alt_c2", "sx_alt_c3", "alternating cycle 2 vs 3"),
    ("sx_norm_t60", "sx_hist", "normalized TPE 60 vs historical reference"),
    ("sx_gap48_t30", "sx_hist", "gap-robust TPE 30 vs historical reference"),
)
SHORT_NAMES = {
    "sx_hist": "HIST",
    "sx_raw_t60": "RAW-T60",
    "sx_norm_t10": "NORM-T10",
    "sx_norm_t30": "NORM-T30",
    "sx_norm_t60": "NORM-T60",
    "sx_random_t30": "RAND-T30",
    "sx_gap48_t30": "GAP-T30",
    "sx_alt_c1": "ALT-C1",
    "sx_alt_c2": "ALT-C2",
    "sx_alt_c3": "ALT-C3",
    "SNaive-24": "SNaive-24",
    "SNaive-168": "SNaive-168",
}


@dataclass(frozen=True)
class DMResult:
    statistic: float
    p_value: float
    n: int
    mean_difference: float
    standard_error: float
    ci_low: float
    ci_high: float
    hac_lag: int
    status: str


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


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_hac_lag(specification: str, n: int) -> int:
    if specification.lower() == "auto":
        lag = int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    else:
        lag = int(specification)
    return max(0, min(lag, max(0, n - 2)))


def dm_test_losses(
    loss_a: Sequence[float],
    loss_b: Sequence[float],
    *,
    hac_lag: str = "auto",
    min_days: int = 20,
    alpha: float = 0.05,
) -> DMResult:
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    differential = a[mask] - b[mask]
    n = int(len(differential))
    if n < int(min_days):
        return DMResult(
            math.nan,
            math.nan,
            n,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            0,
            "insufficient sample",
        )
    mean = float(np.mean(differential))
    lag = _resolve_hac_lag(hac_lag, n)
    if np.allclose(differential, 0.0, rtol=0.0, atol=1e-12):
        return DMResult(0.0, 1.0, n, 0.0, 0.0, 0.0, 0.0, lag, "identical losses")
    centered = differential - mean
    gamma0 = float(np.dot(centered, centered) / n)
    long_run_variance = gamma0
    for current_lag in range(1, lag + 1):
        gamma = float(np.dot(centered[current_lag:], centered[:-current_lag]) / n)
        weight = 1.0 - current_lag / (lag + 1.0)
        long_run_variance += 2.0 * weight * gamma
    if not np.isfinite(long_run_variance) or long_run_variance <= 1e-15:
        return DMResult(
            math.nan,
            math.nan,
            n,
            mean,
            math.nan,
            math.nan,
            math.nan,
            lag,
            "invalid HAC variance",
        )
    standard_error = math.sqrt(long_run_variance / n)
    hln_factor = math.sqrt((n - 1) / n)
    effective_standard_error = standard_error / hln_factor
    statistic = mean / effective_standard_error
    p_value = float(2.0 * student_t.sf(abs(statistic), df=n - 1))
    critical = float(student_t.ppf(1.0 - alpha / 2.0, df=n - 1))
    return DMResult(
        float(statistic),
        p_value,
        n,
        mean,
        float(effective_standard_error),
        float(mean - critical * effective_standard_error),
        float(mean + critical * effective_standard_error),
        lag,
        "ok",
    )


def holm_adjust(values: Sequence[float]) -> np.ndarray:
    p_values = np.asarray(values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(p_values))
    if not len(finite):
        return adjusted
    ordered = finite[np.argsort(p_values[finite])]
    running = 0.0
    count = len(ordered)
    for rank, index in enumerate(ordered):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _paired_series(
    daily: pd.DataFrame,
    *,
    model_a: str,
    model_b: str,
    period: str,
    fit_gap_hours: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]:
    block = daily[
        (daily["period"] == period)
        & (daily["fit_gap_hours"] == int(fit_gap_hours))
        & (daily["snapshot"].isin([model_a, model_b]))
    ]
    pivot = block.pivot_table(
        index="origin",
        columns="snapshot",
        values=PRIMARY_LOSS,
        aggfunc="first",
    ).dropna()
    if model_a not in pivot.columns or model_b not in pivot.columns:
        return np.array([]), np.array([]), []
    return (
        pivot[model_a].to_numpy(dtype=float),
        pivot[model_b].to_numpy(dtype=float),
        [pd.Timestamp(value) for value in pivot.index],
    )


def planned_dm_contrasts(
    daily: pd.DataFrame,
    *,
    period: str = PRIMARY_PERIOD,
    hac_lag: str = "auto",
    min_days: int = 20,
    require_all: bool = True,
) -> pd.DataFrame:
    available = set(daily["snapshot"].astype(str))
    missing = sorted(
        {
            model
            for first, second, _ in PLANNED_CONTRASTS
            for model in (first, second)
            if model not in available
        }
    )
    if require_all and missing:
        raise ValueError(f"Planned DM configurations are missing: {missing}")
    rows: list[dict[str, Any]] = []
    for contrast_no, (first, second, label) in enumerate(
        PLANNED_CONTRASTS, start=1
    ):
        if first not in available or second not in available:
            continue
        loss_a, loss_b, origins = _paired_series(
            daily,
            model_a=first,
            model_b=second,
            period=period,
        )
        result = dm_test_losses(
            loss_a, loss_b, hac_lag=hac_lag, min_days=min_days
        )
        rows.append(
            {
                "contrast_no": contrast_no,
                "label": label,
                "model_a": first,
                "model_b": second,
                "short_a": SHORT_NAMES.get(first, first),
                "short_b": SHORT_NAMES.get(second, second),
                "period": period,
                "loss": PRIMARY_LOSS,
                "first_origin": min(origins) if origins else pd.NaT,
                "last_origin": max(origins) if origins else pd.NaT,
                **asdict(result),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["holm_p"] = holm_adjust(table["p_value"].to_numpy(dtype=float))
    table["significant_0_05"] = table["holm_p"] < 0.05
    table["lower_loss_configuration"] = np.where(
        table["mean_difference"] < 0,
        table["model_a"],
        np.where(table["mean_difference"] > 0, table["model_b"], "tie"),
    )
    return table


def all_pairwise_dm(
    daily: pd.DataFrame,
    *,
    period: str = PRIMARY_PERIOD,
    hac_lag: str = "auto",
    min_days: int = 20,
    excluded_models: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    block = daily[
        (daily["period"] == period) & (daily["fit_gap_hours"] == 0)
    ]
    if excluded_models:
        block = block[~block["snapshot"].isin(set(excluded_models))]
    models = sorted(
        block["snapshot"].unique(),
        key=lambda value: list(SHORT_NAMES).index(value)
        if value in SHORT_NAMES
        else len(SHORT_NAMES),
    )
    rows: list[dict[str, Any]] = []
    for first_index, first in enumerate(models):
        for second in models[first_index + 1 :]:
            loss_a, loss_b, _ = _paired_series(
                daily,
                model_a=first,
                model_b=second,
                period=period,
            )
            result = dm_test_losses(
                loss_a, loss_b, hac_lag=hac_lag, min_days=min_days
            )
            rows.append(
                {
                    "model_a": first,
                    "model_b": second,
                    **asdict(result),
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["holm_p"] = holm_adjust(table["p_value"].to_numpy(dtype=float))
    table["significant_0_05"] = table["holm_p"] < 0.05
    return table


def exact_prediction_duplicates(
    predictions: pd.DataFrame,
    *,
    period: str = PRIMARY_PERIOD,
) -> pd.DataFrame:
    required = {
        "snapshot",
        "period",
        "fit_gap_hours",
        "origin",
        "timestamp",
        "horizon",
        "target",
        "prediction",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(
            f"Prediction table cannot support duplicate detection; missing={missing}"
        )
    block = predictions[
        (predictions["period"] == period)
        & (predictions["fit_gap_hours"] == 0)
    ].copy()
    model_order = {model: index for index, model in enumerate(SHORT_NAMES)}
    models = sorted(
        block["snapshot"].astype(str).unique(),
        key=lambda model: (model_order.get(model, len(model_order)), model),
    )
    canonical_by_signature: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    key_columns = ["origin", "timestamp", "horizon", "target"]
    expected_keys: Optional[pd.DataFrame] = None
    for model in models:
        group = (
            block[block["snapshot"] == model]
            .sort_values(key_columns)
            .reset_index(drop=True)
        )
        keys = group[key_columns]
        if expected_keys is None:
            expected_keys = keys
        elif not keys.equals(expected_keys):
            raise ValueError(
                f"Prediction grid differs for {model}; exact duplicate detection "
                "requires complete aligned P-Q vectors."
            )
        values = group["prediction"].to_numpy(dtype=np.float64)
        digest = hashlib.sha256()
        digest.update(
            pd.util.hash_pandas_object(keys, index=False)
            .to_numpy(dtype=np.uint64, copy=False)
            .tobytes()
        )
        digest.update(values.tobytes())
        signature = digest.hexdigest()
        canonical = canonical_by_signature.setdefault(signature, model)
        rows.append(
            {
                "snapshot": model,
                "canonical_snapshot": canonical,
                "is_exact_duplicate": model != canonical,
                "prediction_sha256": signature,
                "prediction_rows": int(len(group)),
                "period": period,
                "fit_gap_hours": 0,
            }
        )
    return pd.DataFrame(rows)


def _plot_pairwise_heatmap(
    pairwise: pd.DataFrame, path: Path, *, title: str
) -> None:
    if pairwise.empty:
        return
    models = list(
        dict.fromkeys(
            [*pairwise["model_a"].tolist(), *pairwise["model_b"].tolist()]
        )
    )
    labels = [SHORT_NAMES.get(model, model) for model in models]
    n = len(models)
    values = np.ones((n, n), dtype=float)
    directions = np.zeros((n, n), dtype=float)
    annotations = np.full((n, n), "-", dtype=object)
    index = {model: position for position, model in enumerate(models)}
    for row in pairwise.itertuples(index=False):
        i, j = index[row.model_a], index[row.model_b]
        p_value = float(row.holm_p)
        mean = float(row.mean_difference)
        values[i, j] = values[j, i] = p_value
        directions[i, j] = 1.0 if mean < 0 else -1.0
        directions[j, i] = -directions[i, j]
        text = "<.001" if p_value < 0.001 else f"{p_value:.3f}"
        annotations[i, j] = annotations[j, i] = text
    color = np.where(values < 0.05, directions, 0.0)
    np.fill_diagonal(color, np.nan)
    sns.set_theme(style="white", context="paper")
    width = max(7.0, 0.58 * n + 2.4)
    fig, ax = plt.subplots(figsize=(width, width * 0.82))
    sns.heatmap(
        color,
        annot=annotations,
        fmt="",
        cmap=sns.color_palette(["#d97904", "#eeeeee", "#2474a6"], as_cmap=True),
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.7,
        linecolor="white",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Column configuration")
    ax.set_ylabel("Row configuration")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.text(
        0.5,
        0.01,
        "Cell: Holm-adjusted p-value. Blue: row has lower mean loss; "
        "orange: row has higher mean loss; gray: not significant.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_planned_forest(table: pd.DataFrame, path: Path) -> None:
    if table.empty:
        return
    plot = table.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot))
    mean = plot["mean_difference"].to_numpy(dtype=float)
    low = plot["ci_low"].to_numpy(dtype=float)
    high = plot["ci_high"].to_numpy(dtype=float)
    colors = np.where(plot["holm_p"] < 0.05, "#1f6f8b", "#777777")
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.axvline(0.0, color="black", linewidth=0.9)
    for position in range(len(plot)):
        ax.errorbar(
            mean[position],
            y[position],
            xerr=[
                [mean[position] - low[position]],
                [high[position] - mean[position]],
            ],
            fmt="o",
            color=colors[position],
            capsize=3,
        )
    labels = [
        f"{row.short_a} vs {row.short_b}" for row in plot.itertuples(index=False)
    ]
    ax.set_yticks(y, labels)
    ax.set_xlabel("Mean daily paired-loss difference (first minus second)")
    ax.set_title(
        "Planned daily Diebold-Mariano contrasts", loc="left", fontweight="bold"
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _load_search_anytime(
    package_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    shared_run_ids = (
        "sx_raw_t60",
        "sx_norm_t60",
        "sx_random_t30",
        "sx_gap48_t30",
    )
    for run_id in shared_run_ids:
        directory = package_root / "_work" / "search" / run_id
        trials_path = directory / "trials.csv"
        if not trials_path.exists():
            continue
        frame = pd.read_csv(trials_path)
        keep = frame["state"].isin(["COMPLETE", "PRUNED"])
        if "duplicate_parameter_vector" in frame:
            duplicate = frame["duplicate_parameter_vector"]
            if duplicate.dtype != bool:
                duplicate = duplicate.astype(str).str.lower().eq("true")
            keep &= ~duplicate
        frame = frame[keep].copy()
        if frame.empty:
            continue
        frame.sort_values("trial_number", inplace=True)
        frame["fit_count"] = pd.to_numeric(
            frame["fit_count"], errors="coerce"
        ).fillna(0)
        frame["duration_seconds"] = pd.to_numeric(
            frame["duration_seconds"], errors="coerce"
        ).fillna(0.0)
        frame["cumulative_target_fold_fits"] = frame["fit_count"].cumsum()
        frame["cumulative_wall_time_seconds"] = frame[
            "duration_seconds"
        ].cumsum()
        best = math.inf
        best_values: list[float] = []
        for trial in frame.itertuples(index=False):
            objective = float(trial.objective)
            if trial.state == "COMPLETE" and np.isfinite(objective):
                best = min(best, objective)
            best_values.append(best if np.isfinite(best) else math.nan)
        frame["best_complete_objective"] = best_values
        frame["run_id"] = directory.name
        frame["series_kind"] = "shared_trial"
        rows.append(frame)
        complete = frame[frame["state"] == "COMPLETE"]
        summaries.append(
            {
                "run_id": directory.name,
                "objective_kind": str(frame["objective_kind"].iloc[0]),
                "terminal_nonduplicate_trials": int(len(frame)),
                "complete_trials": int(len(complete)),
                "pruned_trials": int((frame["state"] == "PRUNED").sum()),
                "target_fold_fits": int(frame["fit_count"].sum()),
                "trial_duration_seconds": float(frame["duration_seconds"].sum()),
                "best_complete_objective": (
                    float(complete["objective"].min())
                    if not complete.empty
                    else math.nan
                ),
                "best_stored_normalized_objective": (
                    float(
                        pd.to_numeric(
                            complete["normalized_objective"], errors="coerce"
                        ).min()
                    )
                    if not complete.empty
                    and "normalized_objective" in complete
                    else math.nan
                ),
            }
        )

    cycle_path = (
        package_root
        / "_work"
        / "search"
        / "sx_alt_joint"
        / "cycle_costs.csv"
    )
    if cycle_path.exists():
        cycles = pd.read_csv(cycle_path)
        if not cycles.empty:
            cycles = cycles.sort_values("cycle").copy()
            cycles["run_id"] = "sx_alt_joint"
            cycles["series_kind"] = "alternating_cycle"
            cycles["objective_kind"] = "normalized"
            cycles["trial_number"] = cycles["cycle"]
            cycles["state"] = "COMPLETE"
            cycles["objective"] = cycles["normalized_objective"]
            cycles["best_complete_objective"] = cycles[
                "normalized_objective"
            ].cummin()
            cycles["fit_count"] = cycles["total_target_fits"]
            cycles["duration_seconds"] = cycles["wall_time_seconds"]
            rows.append(cycles)
            summaries.append(
                {
                    "run_id": "sx_alt_joint",
                    "objective_kind": "normalized",
                    "terminal_nonduplicate_trials": int(len(cycles)),
                    "complete_trials": int(len(cycles)),
                    "pruned_trials": 0,
                    "target_fold_fits": int(cycles["total_target_fits"].sum()),
                    "trial_duration_seconds": float(
                        cycles["wall_time_seconds"].sum()
                    ),
                    "best_complete_objective": float(
                        cycles["normalized_objective"].min()
                    ),
                    "best_stored_normalized_objective": float(
                        cycles["normalized_objective"].min()
                    ),
                }
            )
    return (
        pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(),
        pd.DataFrame(summaries),
    )


def _plot_search_curves(
    data: pd.DataFrame,
    path: Path,
    *,
    x_column: str = "cumulative_target_fold_fits",
    x_label: str = "Cumulative target-fold fits",
) -> None:
    if data.empty:
        return
    panels = (
        (("sx_raw_t60",), "SEARCH-24 raw objective"),
        (("sx_norm_t60", "sx_random_t30"), "SEARCH-24 normalized objective"),
        (("sx_gap48_t30",), "GAP-48 normalized objective"),
        (("sx_alt_joint",), "Alternating complete-pair audits"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6))
    for ax, (run_ids, panel_title) in zip(axes.ravel(), panels):
        panel = data[data["run_id"].isin(run_ids)]
        for run_id, group in panel.groupby("run_id", sort=True):
            group = group.sort_values(x_column)
            ax.step(
                group[x_column],
                group["best_complete_objective"],
                where="post",
                label=(
                    "ALT"
                    if run_id == "sx_alt_joint"
                    else SHORT_NAMES.get(run_id, run_id)
                ),
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel("Best objective so far")
        ax.set_title(panel_title, loc="left", fontweight="bold")
        ax.grid(alpha=0.25)
        if not panel.empty:
            ax.legend(frameon=False)
        else:
            ax.text(0.5, 0.5, "Not available", ha="center", va="center")
    fig.suptitle(
        "Cost-aware search convergence", x=0.02, ha="left", fontweight="bold"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_gap_robustness(summary: pd.DataFrame, path: Path) -> None:
    retained = {
        "sx_hist",
        "sx_raw_t60",
        "sx_norm_t60",
        "sx_gap48_t30",
        "sx_alt_c3",
    }
    block = summary[
        (summary["period"] == "event16_gaps")
        & (summary["target"] == "PAIRED")
        & (summary["snapshot"].isin(retained))
    ].copy()
    if block.empty:
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for snapshot, group in block.groupby("snapshot", sort=True):
        group = group.sort_values("fit_gap_hours")
        ax.plot(
            group["fit_gap_hours"],
            group["joint_normalized_RMSE"],
            marker="o",
            label=SHORT_NAMES.get(snapshot, snapshot),
        )
    ax.set_xlabel("Estimator-fitting gap (h)")
    ax.set_ylabel("Joint normalized RMSE")
    ax.set_title(
        "Event performance under stale estimator fitting",
        loc="left",
        fontweight="bold",
    )
    ax.set_xticks([0, 24, 72, 168])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _milestone_summary(package_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    models_dir = package_root / "Models"
    for path, payload in model_registry.iter_metadata(models_dir):
        if str(payload.get("target", "")) != "P_Power":
            continue
        snapshot = str(payload.get("experiment_snapshot", ""))
        if snapshot not in SHORT_NAMES or snapshot == "sx_hist":
            continue
        source = payload.get("source", {})
        rows.append(
            {
                "snapshot": snapshot,
                "short_name": SHORT_NAMES.get(snapshot, snapshot),
                "strategy": source.get("strategy", "shared"),
                "terminal_milestone": source.get("terminal_milestone"),
                "cycle": source.get("cycle"),
                "study_name": source.get("study_name"),
                "selected_trial_number": source.get("trial_number"),
                "inner_objective": source.get(
                    "objective", source.get("normalized_objective")
                ),
                "objective_kind": source.get("objective_kind"),
                "design": source.get("design"),
                "trial_target_fold_fits": source.get(
                    "trial_target_fold_fits"
                ),
                "P_parameter_hash": source.get("P_parameter_hash"),
                "Q_parameter_hash": source.get("Q_parameter_hash"),
            }
        )
    return pd.DataFrame(rows)


def _plot_pq_rmse(
    summary: pd.DataFrame,
    path: Path,
    *,
    period: str = PRIMARY_PERIOD,
    title: str = "Prewar external P-Q accuracy",
) -> None:
    block = summary[
        (summary["period"] == period)
        & (summary["fit_gap_hours"] == 0)
        & (summary["target"].isin(["P_Power", "Q_Power"]))
    ]
    pivot = block.pivot_table(
        index="snapshot", columns="target", values="RMSE", aggfunc="first"
    ).dropna()
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    ax.scatter(
        pivot["P_Power"],
        pivot["Q_Power"],
        color="#2474a6",
        edgecolor="white",
        linewidth=0.6,
        s=52,
    )
    label_offsets = {
        "sx_hist": (8, -18),
        "sx_norm_t10": (7, 8),
        "sx_norm_t30": (12, 12),
        "sx_norm_t60": (8, -12),
        "SNaive-24": (5, 4),
        "SNaive-168": (5, 4),
    }
    for snapshot, row in pivot.iterrows():
        ax.annotate(
            SHORT_NAMES.get(snapshot, snapshot),
            (row["P_Power"], row["Q_Power"]),
            xytext=label_offsets.get(snapshot, (5, 4)),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Active-power RMSE (kW)")
    ax.set_ylabel("Reactive-power RMSE (kVAr)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_norm_budget(summary: pd.DataFrame, path: Path) -> None:
    budgets = {
        "sx_norm_t10": 10,
        "sx_norm_t30": 30,
        "sx_norm_t60": 60,
    }
    block = summary[
        (summary["period"] == PRIMARY_PERIOD)
        & (summary["fit_gap_hours"] == 0)
        & (summary["target"] == "PAIRED")
        & (summary["snapshot"].isin(budgets))
    ].copy()
    if block.empty:
        return
    block["terminal_trial_budget"] = block["snapshot"].map(budgets)
    block.sort_values("terminal_trial_budget", inplace=True)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(
        block["terminal_trial_budget"],
        block["joint_normalized_RMSE"],
        marker="o",
        color="#2474a6",
    )
    ax.set_xticks([10, 30, 60])
    ax.set_xlabel("Unique terminal-trial milestone")
    ax.set_ylabel("External joint normalized RMSE")
    ax.set_title(
        "Normalized-search budget milestones", loc="left", fontweight="bold"
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_alternating_cycles(anytime: pd.DataFrame, path: Path) -> None:
    block = anytime[anytime["run_id"] == "sx_alt_joint"].copy()
    if block.empty:
        return
    block.sort_values("trial_number", inplace=True)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(
        block["trial_number"],
        block["objective"],
        marker="o",
        color="#7a5195",
    )
    ax.set_xticks(block["trial_number"].astype(int))
    ax.set_xlabel("Completed P-Q alternating cycle")
    ax.set_ylabel("Complete-pair normalized objective")
    ax.set_title(
        "Alternating-search cycle history", loc="left", fontweight="bold"
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_split_design(package_root: Path, path: Path) -> None:
    input_dir = package_root / "Input"
    search = pd.read_excel(input_dir / "splits_search24.xlsx", sheet_name="24")
    gap = pd.read_excel(input_dir / "splits_gap48.xlsx", sheet_name="24")
    search["test_start"] = pd.to_datetime(search["test_start"])
    gap["test_start"] = pd.to_datetime(gap["test_start"])
    colors = {
        "regular": "#2474a6",
        "calendar": "#7a5195",
        "stress": "#d97904",
    }
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 5.8), sharex=False)
    for stratum, group in search.groupby("stratum", sort=False):
        axes[0].scatter(
            group["test_start"],
            np.full(len(group), stratum),
            label=stratum.capitalize(),
            color=colors.get(stratum, "#777777"),
            s=34,
        )
    axes[0].set_title(
        "(a) SEARCH-24 tuning origins", loc="left", fontweight="bold"
    )
    axes[0].set_xlabel("Forecast origin")
    axes[0].set_ylabel("Stratum")
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].legend(frameon=False, ncol=3)

    for stratum, group in gap.groupby("stratum", sort=False):
        axes[1].scatter(
            group["test_start"],
            group["fit_gap_hours"],
            label=stratum.capitalize(),
            color=colors.get(stratum, "#777777"),
            s=30,
        )
    axes[1].set_title(
        "(b) GAP-48 origin-by-fitting-gap design",
        loc="left",
        fontweight="bold",
    )
    axes[1].set_xlabel("Forecast origin")
    axes[1].set_ylabel("Estimator-fitting gap (h)")
    axes[1].set_yticks([0, 24, 72, 168])
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _combined_2022_tables(
    package_root: Path,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    block = predictions[
        predictions["period"].isin(FULL_2022_COMPONENTS)
        & (predictions["fit_gap_hours"] == 0)
    ].copy()
    if block.empty or not set(FULL_2022_COMPONENTS).issubset(
        set(block["period"].astype(str))
    ):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    block["period"] = FULL_2022_PERIOD
    frame = evaluator.core._read_pq_sheet_xlsx(
        package_root / "Input" / "PQ.xlsx",
        evaluator.core.EXPERIMENT_SHEET,
    )
    scale_payload = _read_json(
        package_root / "Input" / "objective_scales.json", {}
    )
    scales = {
        target: float(scale_payload["target_scales"][target])
        for target in evaluator.TARGETS
    }
    summary_parts: list[pd.DataFrame] = []
    daily_parts: list[pd.DataFrame] = []
    for snapshot, group in block.groupby("snapshot", sort=True):
        summary, daily = evaluator._summarize_predictions(
            frame=frame,
            predictions=group,
            snapshot=str(snapshot),
            period=FULL_2022_PERIOD,
            scales=scales,
        )
        summary_parts.append(summary)
        daily_parts.append(daily)
    return (
        block,
        pd.concat(summary_parts, ignore_index=True),
        pd.concat(daily_parts, ignore_index=True),
    )


def _plot_2022_forecasts(predictions: pd.DataFrame, path: Path) -> None:
    if predictions.empty:
        return
    model_order = ["sx_hist", "sx_norm_t10", "sx_norm_t30", "sx_norm_t60"]
    colors = {
        "sx_hist": "#6b7280",
        "sx_norm_t10": "#2a9d8f",
        "sx_norm_t30": "#e76f51",
        "sx_norm_t60": "#2474a6",
    }
    labels = {
        "sx_hist": "HIST",
        "sx_norm_t10": "NORM-T10",
        "sx_norm_t30": "NORM-T30",
        "sx_norm_t60": "NORM-T60",
    }
    block = predictions[predictions["snapshot"].isin(model_order)].copy()
    if block.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10.0, 5.8), sharex=True)
    targets = (
        ("P_Power", "Active power P", "kW"),
        ("Q_Power", "Reactive power Q", "kVAr"),
    )
    stress_start = pd.Timestamp("2022-02-24 00:00")
    stress_end = pd.Timestamp("2022-03-01 00:00")
    for ax, (target, title, unit) in zip(axes, targets):
        target_block = block[block["target"] == target]
        actual = (
            target_block[["timestamp", "actual"]]
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
        )
        ax.plot(
            actual["timestamp"],
            actual["actual"],
            color="#111827",
            linewidth=1.0,
            label="Observed",
            zorder=5,
        )
        for snapshot in model_order:
            model = target_block[target_block["snapshot"] == snapshot].sort_values(
                "timestamp"
            )
            if model.empty:
                continue
            ax.plot(
                model["timestamp"],
                model["prediction"],
                color=colors[snapshot],
                linewidth=0.75,
                alpha=0.9,
                linestyle=":" if snapshot == "sx_hist" else "-",
                label=labels[snapshot],
            )
        ax.axvspan(
            stress_start,
            stress_end,
            color="#d1d5db",
            alpha=0.35,
            linewidth=0,
        )
        ax.set_ylabel(unit)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Forecast timestamp")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle(
        "Rolling day-ahead forecasts, 1 January-28 February 2022",
        x=0.02,
        y=0.995,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_2022_horizon_rmse(predictions: pd.DataFrame, path: Path) -> None:
    if predictions.empty:
        return
    model_order = [
        "sx_hist",
        "sx_norm_t10",
        "sx_norm_t30",
        "sx_norm_t60",
        "SNaive-24",
        "SNaive-168",
    ]
    colors = {
        "sx_hist": "#6b7280",
        "sx_norm_t10": "#2a9d8f",
        "sx_norm_t30": "#e76f51",
        "sx_norm_t60": "#2474a6",
        "SNaive-24": "#9467bd",
        "SNaive-168": "#8c564b",
    }
    block = predictions[predictions["snapshot"].isin(model_order)].copy()
    if block.empty:
        return
    horizon = (
        block.groupby(["target", "snapshot", "horizon"], as_index=False)["error"]
        .agg(lambda values: float(np.sqrt(np.mean(np.asarray(values) ** 2))))
        .rename(columns={"error": "RMSE"})
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharex=True)
    targets = (
        ("P_Power", "Active power P", "RMSE (kW)"),
        ("Q_Power", "Reactive power Q", "RMSE (kVAr)"),
    )
    for ax, (target, title, ylabel) in zip(axes, targets):
        target_block = horizon[horizon["target"] == target]
        for snapshot in model_order:
            model = target_block[target_block["snapshot"] == snapshot]
            if model.empty:
                continue
            ax.plot(
                model["horizon"],
                model["RMSE"],
                color=colors[snapshot],
                linewidth=1.5,
                linestyle="--" if snapshot.startswith("SNaive") else "-",
                label=SHORT_NAMES.get(snapshot, snapshot),
            )
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.set_xlabel("Forecast horizon (h)")
        ax.set_ylabel(ylabel)
        ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
        ax.grid(alpha=0.25)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.93),
    )
    fig.suptitle(
        "Horizon-wise accuracy, 1 January-28 February 2022",
        x=0.02,
        y=0.995,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run_analysis(
    *,
    package_root: Path,
    require_all: bool,
    hac_lag: str,
    min_days: int,
) -> Dict[str, Any]:
    forecasting_root = package_root / "_work" / "forecasting"
    daily_path = forecasting_root / "daily_paired_losses.csv"
    summary_path = forecasting_root / "metrics_summary.csv"
    predictions_path = forecasting_root / "predictions.csv"
    if (
        not daily_path.exists()
        or not summary_path.exists()
        or not predictions_path.exists()
    ):
        raise FileNotFoundError(
            "Production evaluation outputs are missing. Run evaluate_search.py "
            "without --max-origins before statistical analysis."
        )
    daily = pd.read_csv(daily_path, parse_dates=["origin"])
    summary = pd.read_csv(summary_path)
    predictions = pd.read_csv(
        predictions_path, parse_dates=["origin", "timestamp"]
    )
    combined_predictions, combined_summary, combined_daily = (
        _combined_2022_tables(package_root, predictions)
    )
    planned = planned_dm_contrasts(
        daily,
        period=PRIMARY_PERIOD,
        hac_lag=hac_lag,
        min_days=min_days,
        require_all=require_all,
    )
    duplicates = exact_prediction_duplicates(
        predictions, period=PRIMARY_PERIOD
    )
    excluded = (
        duplicates.loc[
            duplicates["is_exact_duplicate"], "snapshot"
        ].astype(str).tolist()
        if not duplicates.empty
        else []
    )
    pairwise = all_pairwise_dm(
        daily,
        period=PRIMARY_PERIOD,
        hac_lag=hac_lag,
        min_days=min_days,
        excluded_models=excluded,
    )
    combined_planned = (
        planned_dm_contrasts(
            combined_daily,
            period=FULL_2022_PERIOD,
            hac_lag=hac_lag,
            min_days=min_days,
            require_all=False,
        )
        if not combined_daily.empty
        else pd.DataFrame()
    )
    combined_pairwise = (
        all_pairwise_dm(
            combined_daily,
            period=FULL_2022_PERIOD,
            hac_lag=hac_lag,
            min_days=min_days,
            excluded_models=excluded,
        )
        if not combined_daily.empty
        else pd.DataFrame()
    )
    q4_replication = planned_dm_contrasts(
        daily,
        period="q4_2021",
        hac_lag=hac_lag,
        min_days=min_days,
        require_all=False,
    )
    if not q4_replication.empty:
        q4_replication["interpretation"] = (
            "Out-of-search for newly searched configurations; descriptive for "
            "HIST because its archived tuning schedule included later 2021 folds."
        )
    anytime, search_summary = _load_search_anytime(package_root)
    milestones = _milestone_summary(package_root)
    cycle_history = (
        anytime[anytime["run_id"] == "sx_alt_joint"].copy()
        if not anytime.empty
        else pd.DataFrame()
    )
    pruning_path = (
        package_root
        / "_work"
        / "search"
        / "sx_norm_t60"
        / "pruning_replay"
        / "pruning_replay.csv"
    )
    pruning_audit = (
        pd.read_csv(pruning_path) if pruning_path.exists() else pd.DataFrame()
    )
    statistics_root = package_root / "_work" / "statistics"
    figures_root = package_root / "_work" / "figures"
    tables_root = package_root / "_work" / "tables"
    for directory in (statistics_root, figures_root, tables_root):
        directory.mkdir(parents=True, exist_ok=True)
    planned.to_csv(statistics_root / "planned_dm_contrasts.csv", index=False)
    pairwise.to_csv(statistics_root / "all_pairwise_dm.csv", index=False)
    combined_planned.to_csv(
        statistics_root / "jan_feb_2022_planned_dm.csv", index=False
    )
    combined_pairwise.to_csv(
        statistics_root / "jan_feb_2022_all_pairwise_dm.csv", index=False
    )
    duplicates.to_csv(
        statistics_root / "exact_prediction_duplicates.csv", index=False
    )
    q4_replication.to_csv(
        statistics_root / "q4_replication_dm.csv", index=False
    )
    summary.to_csv(tables_root / "evaluation_metrics.csv", index=False)
    combined_summary.to_csv(
        tables_root / "jan_feb_2022_metrics.csv", index=False
    )
    combined_daily.to_csv(
        package_root
        / "_work"
        / "forecasting"
        / "jan_feb_2022_daily_paired_losses.csv",
        index=False,
    )
    anytime.to_csv(tables_root / "anytime_curves.csv", index=False)
    search_summary.to_csv(tables_root / "search_summary.csv", index=False)
    milestones.to_csv(tables_root / "milestone_summary.csv", index=False)
    cycle_history.to_csv(tables_root / "cycle_history.csv", index=False)
    pruning_audit.to_csv(tables_root / "pruning_audit.csv", index=False)
    _plot_planned_forest(planned, figures_root / "planned_dm_contrasts.png")
    _plot_pairwise_heatmap(
        pairwise,
        figures_root / "all_pairwise_dm_heatmap.png",
        title="Prewar daily paired-loss comparison",
    )
    _plot_pairwise_heatmap(
        combined_pairwise,
        figures_root / "jan_feb_2022_all_pairwise_dm_heatmap.png",
        title="January-February 2022 daily paired-loss comparison",
    )
    _plot_search_curves(
        anytime, figures_root / "search_convergence_by_fits.png"
    )
    _plot_search_curves(
        anytime,
        figures_root / "search_convergence_by_time.png",
        x_column="cumulative_wall_time_seconds",
        x_label="Cumulative recorded wall time (s)",
    )
    _plot_gap_robustness(summary, figures_root / "event_gap_robustness.png")
    _plot_pq_rmse(summary, figures_root / "external_pq_rmse.png")
    _plot_pq_rmse(
        combined_summary,
        figures_root / "jan_feb_2022_pq_rmse.png",
        period=FULL_2022_PERIOD,
        title="January-February 2022 P-Q accuracy",
    )
    _plot_2022_forecasts(
        combined_predictions,
        figures_root / "jan_feb_2022_forecast_trajectories.png",
    )
    _plot_2022_horizon_rmse(
        combined_predictions,
        figures_root / "jan_feb_2022_horizon_rmse.png",
    )
    _plot_norm_budget(
        summary, figures_root / "normalized_trial_budget.png"
    )
    _plot_alternating_cycles(
        anytime, figures_root / "alternating_cycles.png"
    )
    _plot_split_design(package_root, figures_root / "split_design.png")
    with pd.ExcelWriter(tables_root / "search_experiment_tables.xlsx") as writer:
        summary.to_excel(writer, sheet_name="evaluation_metrics", index=False)
        planned.to_excel(writer, sheet_name="planned_DM", index=False)
        pairwise.to_excel(writer, sheet_name="all_pairwise_DM", index=False)
        combined_summary.to_excel(
            writer, sheet_name="jan_feb_metrics", index=False
        )
        combined_planned.to_excel(
            writer, sheet_name="jan_feb_planned_DM", index=False
        )
        combined_pairwise.to_excel(
            writer, sheet_name="jan_feb_pairwise_DM", index=False
        )
        duplicates.to_excel(
            writer, sheet_name="exact_duplicates", index=False
        )
        q4_replication.to_excel(
            writer, sheet_name="Q4_replication_DM", index=False
        )
        search_summary.to_excel(
            writer, sheet_name="search_summary", index=False
        )
        milestones.to_excel(
            writer, sheet_name="milestones", index=False
        )
        if not cycle_history.empty:
            cycle_history.to_excel(
                writer, sheet_name="alternating_cycles", index=False
            )
        if not pruning_audit.empty:
            pruning_audit.to_excel(
                writer, sheet_name="pruning_audit", index=False
            )
    report = {
        "script_version": SCRIPT_VERSION,
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "primary_period": PRIMARY_PERIOD,
        "primary_loss": PRIMARY_LOSS,
        "hac_lag": hac_lag,
        "minimum_days": int(min_days),
        "holm_family_size": int(len(planned)),
        "planned_contrasts_available": int(len(planned)),
        "planned_contrasts_expected": int(len(PLANNED_CONTRASTS)),
        "all_pairwise_holm_family_size": int(len(pairwise)),
        "jan_feb_2022_all_pairwise_holm_family_size": int(
            len(combined_pairwise)
        ),
        "exact_duplicate_configurations_excluded_from_pairwise": excluded,
        "files": {
            "planned_dm": str(
                (statistics_root / "planned_dm_contrasts.csv").resolve()
            ),
            "all_pairwise_dm": str(
                (statistics_root / "all_pairwise_dm.csv").resolve()
            ),
            "exact_prediction_duplicates": str(
                (
                    statistics_root / "exact_prediction_duplicates.csv"
                ).resolve()
            ),
            "q4_replication_dm": str(
                (statistics_root / "q4_replication_dm.csv").resolve()
            ),
            "jan_feb_2022_metrics": str(
                (tables_root / "jan_feb_2022_metrics.csv").resolve()
            ),
            "jan_feb_2022_all_pairwise_dm": str(
                (
                    statistics_root
                    / "jan_feb_2022_all_pairwise_dm.csv"
                ).resolve()
            ),
            "anytime_curves": str(
                (tables_root / "anytime_curves.csv").resolve()
            ),
            "workbook": str(
                (tables_root / "search_experiment_tables.xlsx").resolve()
            ),
        },
    }
    _write_json(package_root / "manifests" / "analysis.json", report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=package_root)
    parser.add_argument("--hac-lag", default="auto")
    parser.add_argument("--min-days", type=int, default=20)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Analyze available configurations before all planned jobs finish.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = _build_parser().parse_args(argv)
    run_analysis(
        package_root=args.package_root.expanduser().resolve(),
        require_all=not args.allow_incomplete,
        hac_lag=str(args.hac_lag),
        min_days=int(args.min_days),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""Audit and freeze every input used by the LightGBM search experiment.

The script is intentionally read-only with respect to the source project.  It
writes normalized audit tables, objective scales, and the historical MUT-L24
anchor into the isolated ``LGBM_variance`` package.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


SCRIPT_VERSION = "search-experiment-1.1"
TARGETS = ("P_Power", "Q_Power")
FIXED_OBJECTIVE_SCALES = {
    "P_Power": 192.00558336970016,
    "Q_Power": 229.7979977161049,
}
SEARCH_PARAMETER_KEYS = (
    "n_estimators",
    "learning_rate",
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
)
EXPECTED_DYNAMIC_FEATURES = tuple(
    f"{target}_{suffix}"
    for target in TARGETS
    for suffix in (
        "lag_1",
        "lag_24",
        "rmean_24",
        "rstd_24",
        "rmean_168",
        "rstd_168",
    )
)
REQUIRED_SPLIT_COLUMNS = (
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "origin_id",
    "design_set",
    "scheme",
    "stratum",
    "cluster_id",
    "event_label",
    "fit_gap_hours",
    "history_policy",
    "pruning_order",
    "optuna_use",
)
EXPECTED_SEARCH_ORIGINS = (
    "2021-02-23",
    "2021-03-01",
    "2021-07-17",
    "2021-07-27",
    "2021-08-24",
    "2021-02-16",
    "2021-04-06",
    "2021-05-02",
    "2021-05-14",
    "2021-09-03",
    "2021-08-23",
    "2021-09-27",
    "2021-02-03",
    "2021-05-01",
    "2021-07-15",
    "2021-03-19",
    "2021-05-03",
    "2021-07-16",
    "2021-03-27",
    "2021-08-05",
    "2021-05-18",
    "2021-08-08",
    "2021-06-11",
    "2021-09-19",
)
EXPECTED_SEARCH36_ADDITIONS = (
    "2021-02-24",
    "2021-03-08",
    "2021-03-12",
    "2021-04-19",
    "2021-06-21",
    "2021-08-04",
    "2021-06-17",
    "2021-06-28",
    "2021-09-30",
    "2021-04-22",
    "2021-06-30",
    "2021-09-07",
)
EXPECTED_SEARCH48_ADDITIONS = (
    "2021-02-22",
    "2021-03-09",
    "2021-05-13",
    "2021-04-10",
    "2021-06-01",
    "2021-07-09",
    "2021-05-21",
    "2021-09-01",
    "2021-08-09",
    "2021-04-21",
    "2021-07-05",
    "2021-08-19",
)
EXPECTED_GAP_ORIGINS = (
    "2021-02-23",
    "2021-03-01",
    "2021-02-16",
    "2021-04-06",
    "2021-05-02",
    "2021-05-14",
    "2021-06-11",
    "2021-08-23",
    "2021-07-17",
    "2021-09-03",
    "2021-08-24",
    "2021-09-27",
)
EXPECTED_EVENT_ORIGINS = (
    "2021-10-13",
    "2021-10-14",
    "2021-10-15",
    "2021-12-01",
    "2021-12-24",
    "2021-12-25",
    "2021-12-26",
    "2021-12-27",
    "2021-12-31",
    "2022-01-01",
    "2022-01-02",
    "2022-01-03",
    "2022-01-06",
    "2022-01-07",
    "2022-01-08",
    "2022-01-14",
)


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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_pq_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    with pd.ExcelFile(path) as workbook:
        _require(
            sheet_name in workbook.sheet_names,
            f"PQ workbook does not contain worksheet {sheet_name!r}; "
            f"available={workbook.sheet_names}",
        )
        frame = pd.read_excel(workbook, sheet_name=sheet_name)
    lower = {str(column).strip().lower(): column for column in frame.columns}
    timestamp_column = next(
        (lower[name] for name in ("datetime", "timestamp", "date") if name in lower),
        None,
    )
    _require(timestamp_column is not None, "PQ worksheet has no timestamp column")
    frame = frame.rename(columns={timestamp_column: "Datetime"})
    frame["Datetime"] = pd.to_datetime(frame["Datetime"], errors="raise")
    frame = frame.set_index("Datetime")
    return frame


def _read_split_sheet(
    path: Path, sheet_name: str, *, require_origin_layer: bool = False
) -> pd.DataFrame:
    with pd.ExcelFile(path) as workbook:
        _require(
            workbook.sheet_names == [sheet_name],
            f"{path.name} must contain only worksheet {sheet_name!r}; "
            f"found={workbook.sheet_names}",
        )
        frame = pd.read_excel(workbook, sheet_name=sheet_name)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    missing = [column for column in REQUIRED_SPLIT_COLUMNS if column not in frame.columns]
    _require(not missing, f"{path.name} is missing columns: {missing}")
    if require_origin_layer:
        _require(
            "origin_layer" in frame.columns,
            f"{path.name} is missing required origin_layer",
        )
    for column in ("train_start", "train_end", "test_start", "test_end"):
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    frame["origin_id"] = pd.to_datetime(frame["origin_id"], errors="raise").dt.strftime(
        "%Y-%m-%d"
    )
    frame["fit_gap_hours"] = pd.to_numeric(
        frame["fit_gap_hours"], errors="raise"
    ).astype(int)
    frame["optuna_use"] = frame["optuna_use"].astype(bool)
    return frame


def _audit_pq(frame: pd.DataFrame, split_frames: Iterable[pd.DataFrame]) -> dict[str, Any]:
    _require(isinstance(frame.index, pd.DatetimeIndex), "PQ index is not datetime")
    _require(frame.index.is_monotonic_increasing, "PQ timestamps are not sorted")
    _require(frame.index.is_unique, "PQ timestamps are not unique")
    expected = pd.date_range(frame.index.min(), frame.index.max(), freq="h")
    _require(frame.index.equals(expected), "PQ worksheet is not a complete hourly grid")
    missing_targets = [target for target in TARGETS if target not in frame.columns]
    _require(not missing_targets, f"PQ worksheet is missing targets: {missing_targets}")
    for target in TARGETS:
        values = pd.to_numeric(frame[target], errors="coerce")
        _require(values.notna().all(), f"{target} contains missing or nonnumeric values")
        _require(np.isfinite(values.to_numpy(dtype=float)).all(), f"{target} is non-finite")
    missing_dynamic = [
        column for column in EXPECTED_DYNAMIC_FEATURES if column not in frame.columns
    ]
    _require(
        not missing_dynamic,
        f"PQ worksheet is missing fixed-architecture dynamic features: {missing_dynamic}",
    )

    all_boundaries: list[pd.Timestamp] = []
    for splits in split_frames:
        for column in ("train_start", "train_end", "test_start", "test_end"):
            all_boundaries.extend(pd.Timestamp(value) for value in splits[column])
    absent = sorted({timestamp for timestamp in all_boundaries if timestamp not in frame.index})
    _require(not absent, f"{len(absent)} split boundaries are absent from PQ")

    earliest_origin = min(
        pd.Timestamp(splits["test_start"].min()) for splits in split_frames
    )
    history_start = earliest_origin - pd.Timedelta(hours=168)
    _require(history_start in frame.index, "Earliest origin lacks 168 hours of target history")
    history = frame.loc[history_start : earliest_origin - pd.Timedelta(hours=1), TARGETS]
    _require(len(history) == 168, "Earliest origin history is not exactly 168 hourly rows")
    _require(history.notna().all().all(), "Earliest 168-hour target history is incomplete")

    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "first_timestamp": frame.index.min(),
        "last_timestamp": frame.index.max(),
        "hourly_grid": True,
        "unique_timestamps": True,
        "targets_complete": True,
        "dynamic_features_present": list(EXPECTED_DYNAMIC_FEATURES),
    }


def _audit_common_split_rules(frame: pd.DataFrame, expected_design: str) -> None:
    _require(
        set(frame["design_set"].astype(str)) == {expected_design},
        f"{expected_design}: unexpected design_set values",
    )
    _require(
        set(frame["history_policy"].astype(str)) == {"through_test_start"},
        f"{expected_design}: history policy must be through_test_start",
    )
    duration = (
        (frame["test_end"] - frame["test_start"]) / pd.Timedelta(hours=1) + 1
    ).astype(int)
    _require((duration == 24).all(), f"{expected_design}: every test must contain 24 hours")
    expected_gap = (
        (frame["test_start"] - frame["train_end"]) / pd.Timedelta(hours=1) - 1
    ).astype(int)
    _require(
        expected_gap.equals(frame["fit_gap_hours"]),
        f"{expected_design}: train_end does not match fit_gap_hours",
    )
    _require(
        (frame["train_start"] <= frame["train_end"]).all()
        and (frame["train_end"] < frame["test_start"]).all()
        and (frame["test_start"] <= frame["test_end"]).all(),
        f"{expected_design}: invalid temporal ordering",
    )


def _audit_search(frame: pd.DataFrame) -> dict[str, Any]:
    _audit_common_split_rules(frame, "SEARCH-24")
    _require(len(frame) == 24, "SEARCH-24 must contain 24 rows")
    _require(frame["test_start"].nunique() == 24, "SEARCH-24 origins must be unique")
    _require(set(frame["fit_gap_hours"]) == {0}, "SEARCH-24 must use zero fitting gap")
    _require(frame["optuna_use"].all(), "SEARCH-24 rows must all be enabled for Optuna")
    _require(
        frame["train_start"].eq(pd.Timestamp("2021-01-02 00:00")).all(),
        "SEARCH-24 train_start must be 2021-01-02 00:00",
    )
    _require(
        frame["test_start"].max() <= pd.Timestamp("2021-09-30 00:00"),
        "SEARCH-24 extends beyond September 2021",
    )
    actual_order = pd.to_numeric(frame["pruning_order"], errors="coerce")
    _require(
        np.array_equal(actual_order.to_numpy(dtype=int), np.arange(1, 25)),
        "SEARCH-24 pruning_order is not 1..24",
    )
    counts = frame["stratum"].value_counts().to_dict()
    _require(
        counts == {"regular": 12, "calendar": 6, "stress": 6},
        f"SEARCH-24 stratum counts are wrong: {counts}",
    )
    origins = tuple(frame["origin_id"].astype(str))
    _require(
        origins == EXPECTED_SEARCH_ORIGINS,
        "SEARCH-24 origins or pruning order differ from the frozen design",
    )
    first_twelve = frame.iloc[:12]["stratum"].value_counts().to_dict()
    _require(
        first_twelve == {"regular": 4, "calendar": 4, "stress": 4},
        f"SEARCH-24 first 12 rows are not stratum-balanced: {first_twelve}",
    )
    return {
        "rows": 24,
        "unique_origins": 24,
        "origins_in_pruning_order": list(origins),
        "strata": counts,
        "first_12_strata": first_twelve,
        "fit_gap_hours": {"0": 24},
        "pruning_order": [int(value) for value in actual_order],
    }


def _audit_nested_search(
    frame: pd.DataFrame,
    *,
    parent: pd.DataFrame,
    design: str,
    parent_design: str,
    expected_rows: int,
    expected_additions: tuple[str, ...],
    expected_strata: Mapping[str, int],
    expected_layers: Mapping[str, int],
) -> dict[str, Any]:
    _audit_common_split_rules(frame, design)
    _require(len(frame) == expected_rows, f"{design} must contain {expected_rows} rows")
    _require(
        frame["test_start"].nunique() == expected_rows,
        f"{design} origins must be unique",
    )
    _require(set(frame["fit_gap_hours"]) == {0}, f"{design} must use zero fitting gap")
    _require(frame["optuna_use"].all(), f"{design} rows must all be enabled for Optuna")
    _require(
        frame["train_start"].eq(pd.Timestamp("2021-01-02 00:00")).all(),
        f"{design} train_start must be 2021-01-02 00:00",
    )
    _require(
        frame["test_start"].max() <= pd.Timestamp("2021-09-30 00:00"),
        f"{design} extends beyond September 2021",
    )
    actual_order = pd.to_numeric(frame["pruning_order"], errors="coerce")
    _require(
        np.array_equal(
            actual_order.to_numpy(dtype=int), np.arange(1, expected_rows + 1)
        ),
        f"{design} pruning_order is not 1..{expected_rows}",
    )
    counts = frame["stratum"].value_counts().to_dict()
    _require(
        counts == dict(expected_strata),
        f"{design} stratum counts are wrong: {counts}",
    )
    layers = frame["origin_layer"].value_counts().to_dict()
    _require(
        layers == dict(expected_layers),
        f"{design} origin-layer counts are wrong: {layers}",
    )

    ignored = {"design_set", "origin_layer", "pruning_order"}
    identity_columns = [
        column
        for column in REQUIRED_SPLIT_COLUMNS
        if column not in ignored
    ]
    inherited = frame.iloc[: len(parent)].reset_index(drop=True)
    expected_parent = parent.reset_index(drop=True)
    pd.testing.assert_frame_equal(
        inherited.loc[:, identity_columns],
        expected_parent.loc[:, identity_columns],
        check_dtype=False,
        check_like=False,
    )
    additions = tuple(frame.iloc[len(parent) :]["origin_id"].astype(str))
    _require(
        additions == expected_additions,
        f"{design} additions or append order differ from the frozen design",
    )
    _require(
        not set(additions).intersection(set(parent["origin_id"].astype(str))),
        f"{design} additions overlap inherited {parent_design} origins",
    )
    return {
        "rows": expected_rows,
        "unique_origins": expected_rows,
        "parent_design": parent_design,
        "inherited_rows": int(len(parent)),
        "inherited_identity_columns": identity_columns,
        "ignored_nested_identity_columns": sorted(ignored),
        "additions_in_pruning_order": list(additions),
        "strata": {key: int(value) for key, value in counts.items()},
        "origin_layers": {key: int(value) for key, value in layers.items()},
        "pruning_order": [int(value) for value in actual_order],
    }


def _audit_gap(frame: pd.DataFrame) -> dict[str, Any]:
    _audit_common_split_rules(frame, "GAP-48")
    _require(len(frame) == 48, "GAP-48 must contain 48 rows")
    _require(frame["test_start"].nunique() == 12, "GAP-48 must contain 12 origins")
    _require(frame["optuna_use"].all(), "GAP-48 rows must all be enabled for Optuna")
    actual_order = pd.to_numeric(frame["pruning_order"], errors="coerce")
    _require(
        np.array_equal(actual_order.to_numpy(dtype=int), np.arange(1, 49)),
        "GAP-48 pruning_order is not 1..48",
    )

    gap_counts = frame["fit_gap_hours"].value_counts().sort_index().to_dict()
    _require(
        gap_counts == {0: 12, 24: 12, 72: 12, 168: 12},
        f"GAP-48 gap counts are wrong: {gap_counts}",
    )
    stratum_counts = frame["stratum"].value_counts().to_dict()
    _require(
        stratum_counts == {"regular": 16, "calendar": 16, "stress": 16},
        f"GAP-48 stratum counts are wrong: {stratum_counts}",
    )
    cells = (
        frame.groupby(["fit_gap_hours", "stratum"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    _require((cells == 4).all().all(), f"GAP-48 gap-stratum cells are not all size 4:\n{cells}")

    for origin, group in frame.groupby("origin_id", sort=False):
        _require(len(group) == 4, f"GAP-48 origin {origin} does not have four rows")
        _require(
            set(group["fit_gap_hours"]) == {0, 24, 72, 168},
            f"GAP-48 origin {origin} does not contain all fitting gaps",
        )
        _require(
            group["test_start"].nunique() == 1 and group["test_end"].nunique() == 1,
            f"GAP-48 origin {origin} changes the scored day across gaps",
        )
        _require(
            group["train_start"].nunique() == 1,
            f"GAP-48 origin {origin} changes train_start across gaps",
        )
        _require(
            group["stratum"].nunique() == 1 and group["cluster_id"].nunique() == 1,
            f"GAP-48 origin {origin} changes labels across gaps",
        )

    unique_origins = tuple(dict.fromkeys(frame["origin_id"].astype(str)))
    _require(
        unique_origins == EXPECTED_GAP_ORIGINS,
        "GAP-48 origins or first-occurrence order differ from the frozen design",
    )
    first_sixteen = frame.iloc[:16]
    first_sixteen_gaps = (
        first_sixteen["fit_gap_hours"].value_counts().sort_index().to_dict()
    )
    _require(
        first_sixteen_gaps == {0: 4, 24: 4, 72: 4, 168: 4},
        f"GAP-48 first 16 rows are not gap-balanced: {first_sixteen_gaps}",
    )
    _require(
        set(first_sixteen["stratum"]) == {"regular", "calendar", "stress"},
        "GAP-48 first 16 rows do not represent every stratum",
    )
    return {
        "rows": 48,
        "unique_origins": 12,
        "origins_in_first_occurrence_order": list(unique_origins),
        "gap_counts": {str(key): int(value) for key, value in gap_counts.items()},
        "stratum_counts": {key: int(value) for key, value in stratum_counts.items()},
        "gap_stratum_cell_size": 4,
        "first_16_gap_counts": {
            str(key): int(value) for key, value in first_sixteen_gaps.items()
        },
        "first_16_all_strata_represented": True,
    }


def _audit_event(
    frame: pd.DataFrame, searches: Mapping[str, pd.DataFrame]
) -> dict[str, Any]:
    _audit_common_split_rules(frame, "EVENT-16")
    _require(len(frame) == 16, "EVENT-16 must contain 16 rows")
    _require(frame["test_start"].nunique() == 16, "EVENT-16 origins must be unique")
    _require((~frame["optuna_use"]).all(), "EVENT-16 must be disabled for Optuna")
    _require(set(frame["fit_gap_hours"]) == {0}, "EVENT-16 must use zero fitting gap")
    _require(
        pd.Timestamp("2022-01-01 00:00") in set(frame["test_start"]),
        "EVENT-16 must include 2022-01-01",
    )
    for design, search in searches.items():
        overlap = set(frame["test_start"]).intersection(set(search["test_start"]))
        _require(not overlap, f"EVENT-16 overlaps {design}: {sorted(overlap)}")
    origins = tuple(frame["origin_id"].astype(str))
    _require(
        origins == EXPECTED_EVENT_ORIGINS,
        "EVENT-16 origins or order differ from the frozen design",
    )
    return {
        "rows": 16,
        "unique_origins": 16,
        "origins_in_order": list(origins),
        "includes_2022_01_01": True,
        "disjoint_from_search_designs": sorted(searches),
        "optuna_use": False,
    }


def _pooled_snaive_rmse(
    frame: pd.DataFrame, splits: pd.DataFrame, lag: int = 24
) -> dict[str, float]:
    errors: dict[str, list[float]] = {target: [] for target in TARGETS}
    for _, split in splits.iterrows():
        index = pd.date_range(split["test_start"], split["test_end"], freq="h")
        source = index - pd.Timedelta(hours=lag)
        for target in TARGETS:
            actual = frame[target].reindex(index).to_numpy(dtype=float)
            forecast = frame[target].reindex(source).to_numpy(dtype=float)
            _require(
                np.isfinite(actual).all() and np.isfinite(forecast).all(),
                f"SNaive-{lag} diagnostic is unavailable for {target}",
            )
            errors[target].extend((forecast - actual).tolist())
    return {
        target: float(np.sqrt(np.mean(np.square(values))))
        for target, values in errors.items()
    }


def _seasonal_naive_scales(
    frame: pd.DataFrame, search: pd.DataFrame, lag: int = 24
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    all_errors: dict[str, list[float]] = {target: [] for target in TARGETS}
    for split_no, split in search.iterrows():
        index = pd.date_range(split["test_start"], split["test_end"], freq="h")
        source = index - pd.Timedelta(hours=lag)
        for target in TARGETS:
            actual = frame[target].reindex(index).to_numpy(dtype=float)
            forecast = frame[target].reindex(source).to_numpy(dtype=float)
            _require(
                np.isfinite(actual).all() and np.isfinite(forecast).all(),
                f"SNaive-{lag} is unavailable for split {split_no}, target {target}",
            )
            error = forecast - actual
            all_errors[target].extend(error.tolist())
            rows.append(
                {
                    "fold_index": int(split_no),
                    "pruning_order": int(split["pruning_order"]),
                    "origin_id": split["origin_id"],
                    "stratum": split["stratum"],
                    "cluster_id": split["cluster_id"],
                    "target": target,
                    "n_hours": int(len(error)),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "mae": float(np.mean(np.abs(error))),
                }
            )
    scales = {
        target: float(np.sqrt(np.mean(np.square(errors))))
        for target, errors in all_errors.items()
    }
    payload = {
        "script_version": SCRIPT_VERSION,
        "definition": (
            "Aggregate hourly RMSE of the previous-day seasonal-naive forecast "
            "over all 24 SEARCH-24 origins (576 hours per target)."
        ),
        "benchmark": "SNaive-24",
        "folds": 24,
        "hours_per_target": 576,
        "P_RMSE": scales["P_Power"],
        "Q_RMSE": scales["Q_Power"],
        "target_scales": scales,
        "units": {"P_Power": "kW", "Q_Power": "kVAr"},
    }
    return payload, pd.DataFrame(rows)


def _load_historical_anchor(
    p_meta_path: Path, q_meta_path: Path, original_splits_path: Path
) -> dict[str, Any]:
    _require(p_meta_path.exists(), f"Historical P metadata not found: {p_meta_path}")
    _require(q_meta_path.exists(), f"Historical Q metadata not found: {q_meta_path}")
    p_meta = json.loads(p_meta_path.read_text(encoding="utf-8"))
    q_meta = json.loads(q_meta_path.read_text(encoding="utf-8"))
    for target, meta in zip(TARGETS, (p_meta, q_meta)):
        _require(meta.get("target") == target, f"Historical metadata target mismatch for {target}")
        _require(str(meta.get("model", "")).upper() == "LGBM", "Historical model is not LGBM")
        _require(meta.get("lag_policy") == "mutual", "Historical model is not mutual-history")
        _require(meta.get("feature_selector") == "all", "Historical model does not use all features")
        missing = [column for column in EXPECTED_DYNAMIC_FEATURES if column not in meta.get("features", [])]
        _require(not missing, f"Historical {target} metadata lacks dynamic features: {missing}")

    p_params = p_meta.get("tuned_params") or p_meta.get("best_params") or {}
    q_params = q_meta.get("tuned_params") or q_meta.get("best_params") or {}
    p_vector = {key: p_params.get(key) for key in SEARCH_PARAMETER_KEYS}
    q_vector = {key: q_params.get(key) for key in SEARCH_PARAMETER_KEYS}
    _require(all(value is not None for value in p_vector.values()), "Historical P vector is incomplete")
    _require(all(value is not None for value in q_vector.values()), "Historical Q vector is incomplete")
    _require(p_vector == q_vector, "Historical P and Q do not share the same tunable vector")

    original_freq = {
        "P_Power": p_params.get("subsample_freq"),
        "Q_Power": q_params.get("subsample_freq"),
    }
    with pd.ExcelFile(original_splits_path) as workbook:
        _require("24" in workbook.sheet_names, "Original split workbook lacks worksheet 24")
        original_fold_count = len(pd.read_excel(workbook, sheet_name="24"))

    return {
        "script_version": SCRIPT_VERSION,
        "identifier": "HIST",
        "manuscript_identifier": "MUT-L24",
        "source_metadata": {
            "P_Power": f"Input/{p_meta_path.name}",
            "Q_Power": f"Input/{q_meta_path.name}",
            "P_sha256": _sha256(p_meta_path),
            "Q_sha256": _sha256(q_meta_path),
        },
        "verified_architecture": {
            "model": "LGBM",
            "worksheet": "24",
            "lag_policy": "mutual",
            "feature_selector": "all",
            "dynamic_features": list(EXPECTED_DYNAMIC_FEATURES),
            "paired_targets": list(TARGETS),
            "shared_tunable_vector": True,
        },
        "historical_tunable_params": p_vector,
        "search_anchor_params": p_vector,
        "fixed_search_params": {
            "subsample_freq": 1,
            "random_state": 42,
        },
        "historical_fixed_param_difference": {
            "subsample_freq_in_archived_models": original_freq,
            "subsample_freq_in_new_search": 1,
            "reason": (
                "subsample_freq is fixed outside the nine-dimensional search space; "
                "the archived value is recorded, while every new job uses the "
                "pre-specified fixed value 1."
            ),
        },
        "original_split_workbook": f"Input/{original_splits_path.name}",
        "original_split_sha256": _sha256(original_splits_path),
        "original_sheet24_rows": int(original_fold_count),
        "provenance_note": (
            "The legacy metadata does not store the tuning split count. The associated "
            "project split workbook contains 123 rows on worksheet 24; this is recorded "
            "as source provenance rather than inferred from the model binary."
        ),
    }


def audit_inputs(
    *,
    package_root: Path,
    pq_path: Path,
    search_path: Path,
    search36_path: Path,
    search48_path: Path,
    gap_path: Path,
    event_path: Path,
    historical_p_meta: Path,
    historical_q_meta: Path,
    original_splits_path: Path,
) -> dict[str, Any]:
    package_root = package_root.resolve()
    output_root = package_root / "Results" / "reproducibility"
    output_root.mkdir(parents=True, exist_ok=True)
    for path in (
        pq_path,
        search_path,
        search36_path,
        search48_path,
        gap_path,
        event_path,
        historical_p_meta,
        historical_q_meta,
        original_splits_path,
    ):
        _require(path.exists(), f"Required input does not exist: {path}")

    search = _read_split_sheet(search_path, "24")
    search36 = _read_split_sheet(search36_path, "24", require_origin_layer=True)
    search48 = _read_split_sheet(search48_path, "24", require_origin_layer=True)
    gap = _read_split_sheet(gap_path, "24")
    event = _read_split_sheet(event_path, "24")
    pq = _read_pq_sheet(pq_path, "24")

    code_root = package_root / "code"
    if (code_root / "_internal").is_dir():
        code_root = code_root / "_internal"

    report = {
        "script_version": SCRIPT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "files": {
            "PQ.xlsx": _sha256(pq_path),
            "splits_search24.xlsx": _sha256(search_path),
            "splits_search36.xlsx": _sha256(search36_path),
            "splits_search48.xlsx": _sha256(search48_path),
            "splits_gap48.xlsx": _sha256(gap_path),
            "splits_event16.xlsx": _sha256(event_path),
            "splits_historical.xlsx": _sha256(original_splits_path),
            "historical_source_mut_l24_P.meta.json": _sha256(
                historical_p_meta
            ),
            "historical_source_mut_l24_Q.meta.json": _sha256(
                historical_q_meta
            ),
        },
        "code_files": {
            name: _sha256(code_root / name)
            for name in (
                "audit_search_inputs.py",
                "build_nested_search_splits.py",
                "build_nested_search_workbooks.mjs",
                "Forecasting_search.py",
                "evaluate_search.py",
                "analyze_search.py",
            )
        },
        "pq": _audit_pq(pq, (search, search36, search48, gap, event)),
        "SEARCH-24": _audit_search(search),
        "SEARCH-36": _audit_nested_search(
            search36,
            parent=search,
            design="SEARCH-36",
            parent_design="SEARCH-24",
            expected_rows=36,
            expected_additions=EXPECTED_SEARCH36_ADDITIONS,
            expected_strata={"regular": 18, "calendar": 9, "stress": 9},
            expected_layers={"CORE24": 24, "EXPANSION_A": 12},
        ),
        "SEARCH-48": _audit_nested_search(
            search48,
            parent=search36,
            design="SEARCH-48",
            parent_design="SEARCH-36",
            expected_rows=48,
            expected_additions=EXPECTED_SEARCH48_ADDITIONS,
            expected_strata={"regular": 24, "calendar": 12, "stress": 12},
            expected_layers={
                "CORE24": 24,
                "EXPANSION_A": 12,
                "EXPANSION_B": 12,
            },
        ),
        "GAP-48": _audit_gap(gap),
        "EVENT-16": _audit_event(
            event,
            {
                "SEARCH-24": search,
                "SEARCH-36": search36,
                "SEARCH-48": search48,
            },
        ),
    }

    scales, scale_folds = _seasonal_naive_scales(pq, search, lag=24)
    for target, expected in FIXED_OBJECTIVE_SCALES.items():
        observed = float(scales["target_scales"][target])
        _require(
            np.isclose(observed, expected, rtol=0.0, atol=1e-12),
            f"Frozen objective scale changed for {target}: {observed} != {expected}",
        )
    report["diagnostic_snaive24_rmse"] = {
        "SEARCH-36": _pooled_snaive_rmse(pq, search36, lag=24),
        "SEARCH-48": _pooled_snaive_rmse(pq, search48, lag=24),
        "used_for_objective_scaling": False,
        "objective_scales_remain_frozen_from": "SEARCH-24",
    }
    scales["input_hashes"] = report["files"]
    scale_path = package_root / "Input" / "objective_scales.json"
    _write_json(scale_path, scales)
    scale_folds.to_csv(output_root / "objective_scale_folds.csv", index=False)

    historical = _load_historical_anchor(
        historical_p_meta, historical_q_meta, original_splits_path
    )
    historical["input_hashes"] = report["files"]
    historical_path = package_root / "Input" / "historical_mut_l24_params.json"
    _write_json(historical_path, historical)

    for label, frame in (
        ("search24", search),
        ("search36", search36),
        ("search48", search48),
        ("gap48", gap),
        ("event16", event),
    ):
        frame.to_csv(output_root / f"{label}_normalized.csv", index=False)
    search36.to_csv(output_root / "search36_rows.csv", index=False)
    search48.to_csv(output_root / "search48_rows.csv", index=False)

    nested_report_path = output_root / "nested_split_audit.json"
    _write_json(
        nested_report_path,
        {
            "script_version": SCRIPT_VERSION,
            "created_utc": report["created_utc"],
            "status": "passed",
            "files": {
                "splits_search24.xlsx": report["files"]["splits_search24.xlsx"],
                "splits_search36.xlsx": report["files"]["splits_search36.xlsx"],
                "splits_search48.xlsx": report["files"]["splits_search48.xlsx"],
            },
            "SEARCH-36": report["SEARCH-36"],
            "SEARCH-48": report["SEARCH-48"],
            "diagnostic_snaive24_rmse": report["diagnostic_snaive24_rmse"],
        },
    )

    report["derived_files"] = {
        "objective_scales": {
            "path": str(scale_path.resolve()),
            "sha256": _sha256(scale_path),
        },
        "historical_anchor": {
            "path": str(historical_path.resolve()),
            "sha256": _sha256(historical_path),
        },
        "nested_split_audit": {
            "path": str(nested_report_path.resolve()),
            "sha256": _sha256(nested_report_path),
        },
    }
    report_path = output_root / "input_audit.json"
    _write_json(report_path, report)
    print(f"[audit passed] {report_path}")
    print(
        "[objective scales] "
        f"P={scales['P_RMSE']:.6f}, Q={scales['Q_RMSE']:.6f}"
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=package_root)
    parser.add_argument("--pq", type=Path, default=package_root / "Input" / "PQ.xlsx")
    parser.add_argument(
        "--search-splits",
        type=Path,
        default=package_root / "Input" / "splits_search24.xlsx",
    )
    parser.add_argument(
        "--search36-splits",
        type=Path,
        default=package_root / "Input" / "splits_search36.xlsx",
    )
    parser.add_argument(
        "--search48-splits",
        type=Path,
        default=package_root / "Input" / "splits_search48.xlsx",
    )
    parser.add_argument(
        "--gap-splits",
        type=Path,
        default=package_root / "Input" / "splits_gap48.xlsx",
    )
    parser.add_argument(
        "--event-splits",
        type=Path,
        default=package_root / "Input" / "splits_event16.xlsx",
    )
    parser.add_argument(
        "--historical-p-meta",
        type=Path,
        default=package_root
        / "Input"
        / "historical_source_mut_l24_P.meta.json",
    )
    parser.add_argument(
        "--historical-q-meta",
        type=Path,
        default=package_root
        / "Input"
        / "historical_source_mut_l24_Q.meta.json",
    )
    parser.add_argument(
        "--original-splits",
        type=Path,
        default=package_root / "Input" / "splits_historical.xlsx",
    )
    return parser


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = _build_parser().parse_args()
    audit_inputs(
        package_root=args.package_root,
        pq_path=args.pq,
        search_path=args.search_splits,
        search36_path=args.search36_splits,
        search48_path=args.search48_splits,
        gap_path=args.gap_splits,
        event_path=args.event_splits,
        historical_p_meta=args.historical_p_meta,
        historical_q_meta=args.historical_q_meta,
        original_splits_path=args.original_splits,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

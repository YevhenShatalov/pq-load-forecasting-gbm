#!/usr/bin/env python3
"""Compare evaluated day-ahead forecasting systems and build article figures.

The script consumes the paired Excel workbooks produced by the model-evaluation
stage. It neither refits models nor changes the source workbooks. Its primary
Diebold-Mariano (DM) analysis uses one loss value per complete 24-hour forecast
trajectory, so the repeated observations are forecast origins rather than
individual hours. Horizon-specific tests are retained as secondary diagnostics.

The workflow validates timestamps and actual values, detects exact prediction
duplicates separately for active and reactive power, applies target-specific
Holm correction, and exports the complete numerical audit. It also reproduces
the diagnostic plots used during model comparison and the compact two-panel
all-versus-all matrix used in the manuscript.
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
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - needed only for the manuscript matrix
    Image = ImageDraw = ImageFont = None

try:
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
except Exception:  # pragma: no cover - PNG output remains available
    ImageReader = canvas = None

try:
    from scipy.stats import t as student_t
except Exception:  # pragma: no cover - normal fallback is retained
    student_t = None


TARGETS = ("P_Power", "Q_Power")
TARGET_SHORT = {"P_Power": "P", "Q_Power": "Q"}
TARGET_NAME = {"P_Power": "Active power P", "Q_Power": "Reactive power Q"}
TARGET_UNIT = {"P_Power": "kW", "Q_Power": "kVAr"}
TARGET_SHEETS = {target: f"{target}_predictions" for target in TARGETS}
TARGET_DAILY_SHEETS = {target: f"{target}_daily_metrics" for target in TARGETS}
LOSS_CODES = ("SE", "AE", "CRPS", "PB")
LOSS_LABELS = {
    "SE": "mean squared error",
    "AE": "mean absolute error",
    "CRPS": "mean CRPS",
    "PB": "mean pinball loss",
}
CORE_COLUMNS = ("ts", "y_true", "y_pred")
QUANTILE_RE = re.compile(r"^Q(?P<level>\d{1,2}(?:\.\d+)?)$", re.IGNORECASE)
PINBALL_RE = re.compile(r"^Pinball_(?P<tau>0(?:\.\d+)?|1(?:\.0+)?)$", re.IGNORECASE)

BUNDLE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BUNDLE_DIR.parent if (BUNDLE_DIR.parent / "Proper Model Comparison").exists() else Path.cwd()


@dataclass(frozen=True)
class RunConfig:
    input_dir: str
    output_dir: str
    start: Optional[str]
    end: Optional[str]
    alignment: str
    expected_hours: int
    alpha: float
    hac_lag: str
    min_days: int
    regime_split: Optional[str]
    include_baselines: bool
    deduplicate: bool
    horizon_dm: bool
    plots: bool
    per_model_plots: bool
    max_models: Optional[int]
    publication_figure: bool
    publication_dpi: int
    selected_system: Optional[str]


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _new_run_directory(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    candidate = root / stem
    counter = 1
    while candidate.exists():
        candidate = root / f"{stem}_{counter}"
        counter += 1
    candidate.mkdir()
    return candidate


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _tau_from_quantile(column: str) -> Optional[float]:
    match = QUANTILE_RE.match(str(column).strip())
    if not match:
        return None
    value = float(match.group("level"))
    return value / 100.0 if value > 1.0 else value


def _quantile_columns(frame: pd.DataFrame) -> List[Tuple[float, str]]:
    parsed: List[Tuple[float, str]] = []
    for column in frame.columns:
        tau = _tau_from_quantile(str(column))
        if tau is not None and 0.0 < tau < 1.0:
            parsed.append((tau, str(column)))
    return sorted(parsed)


def _pinball_columns(frame: pd.DataFrame) -> List[Tuple[float, str]]:
    parsed: List[Tuple[float, str]] = []
    for column in frame.columns:
        match = PINBALL_RE.match(str(column).strip())
        if match:
            tau = float(match.group("tau"))
            if 0.0 < tau < 1.0:
                parsed.append((tau, str(column)))
    return sorted(parsed)


def _normalize_prediction_frame(
    frame: pd.DataFrame,
    *,
    source: Path,
    target: str,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> pd.DataFrame:
    data = frame.copy()
    missing = [column for column in CORE_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"{source.name}/{target}: missing columns {missing}")

    data["ts"] = pd.to_datetime(data["ts"], errors="coerce")
    for column in data.columns:
        if column in {"y_true", "y_pred", "L80", "U80", "L95", "U95", "CRPS"}:
            data[column] = _numeric(data[column])
        elif QUANTILE_RE.match(str(column)) or PINBALL_RE.match(str(column)):
            data[column] = _numeric(data[column])

    if data["ts"].isna().any():
        raise ValueError(f"{source.name}/{target}: invalid timestamps")
    if data["ts"].duplicated().any():
        duplicates = int(data["ts"].duplicated(keep=False).sum())
        raise ValueError(f"{source.name}/{target}: {duplicates} duplicated timestamp rows")
    if data[["y_true", "y_pred"]].isna().any().any():
        raise ValueError(f"{source.name}/{target}: missing actual or point forecast values")

    data = data.sort_values("ts").reset_index(drop=True)
    if start is not None:
        data = data[data["ts"] >= start]
    if end is not None:
        data = data[data["ts"] <= end]
    data = data.reset_index(drop=True)
    if data.empty:
        raise ValueError(f"{source.name}/{target}: no rows in the requested period")

    data["date"] = data["ts"].dt.normalize()
    data["horizon"] = data.groupby("date", sort=True).cumcount() + 1
    return data


def _stored_daily_mase(
    excel_file: pd.ExcelFile,
    *,
    source: Path,
    target: str,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> Optional[pd.Series]:
    """Return the evaluator's training-scaled daily MASE indexed by date."""

    sheet = TARGET_DAILY_SHEETS[target]
    if sheet not in excel_file.sheet_names:
        return None
    data = excel_file.parse(sheet)
    if not {"date", "MASE"}.issubset(data.columns):
        return None
    data = data[["date", "MASE"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["MASE"] = _numeric(data["MASE"])
    if data.isna().any().any():
        raise ValueError(f"{source.name}/{sheet}: invalid date or MASE value")
    if data["date"].duplicated().any():
        raise ValueError(f"{source.name}/{sheet}: duplicated daily metric dates")
    if start is not None:
        data = data[data["date"] >= start.normalize()]
    if end is not None:
        data = data[data["date"] <= end.normalize()]
    return data.set_index("date")["MASE"].sort_index()


def _candidate_workbooks(
    input_dir: Path,
    *,
    include_baselines: bool,
) -> Tuple[List[Path], List[Dict[str, str]]]:
    candidates: List[Path] = []
    skipped: List[Dict[str, str]] = []
    for path in sorted(input_dir.glob("*.xlsx")):
        if path.name.startswith("~$"):
            skipped.append({"file": path.name, "reason": "temporary Excel file"})
            continue
        if not include_baselines and path.stem.lower().startswith("seasonal_naive"):
            skipped.append({"file": path.name, "reason": "baseline excluded by configuration"})
            continue
        try:
            sheets = set(pd.ExcelFile(path).sheet_names)
        except Exception as exc:
            skipped.append({"file": path.name, "reason": f"unreadable workbook: {exc}"})
            continue
        required = set(TARGET_SHEETS.values())
        if not required.issubset(sheets):
            skipped.append({"file": path.name, "reason": "not a paired prediction workbook"})
            continue
        candidates.append(path)
    return candidates, skipped


def _signature_columns(frame: pd.DataFrame) -> List[str]:
    columns = ["ts", "y_true", "y_pred", "L80", "U80", "L95", "U95", "CRPS"]
    columns += [column for _, column in _quantile_columns(frame)]
    columns += [column for _, column in _pinball_columns(frame)]
    return [column for column in columns if column in frame.columns]


def _target_prediction_signature(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    columns = _signature_columns(frame)
    digest.update("|".join(columns).encode("utf-8"))
    values = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy(np.uint64)
    digest.update(values.tobytes())
    return digest.hexdigest()


def _prediction_signature(frames: Mapping[str, pd.DataFrame]) -> str:
    digest = hashlib.sha256()
    for target in TARGETS:
        frame = frames[target]
        digest.update(target.encode("ascii"))
        digest.update(_target_prediction_signature(frame).encode("ascii"))
    return digest.hexdigest()


def _count_quantile_crossings(frame: pd.DataFrame) -> int:
    columns = [column for _, column in _quantile_columns(frame)]
    if len(columns) < 2:
        return 0
    values = frame[columns].to_numpy(float)
    return int(np.any(np.diff(values, axis=1) < -1e-10, axis=1).sum())


def _count_interval_inversions(frame: pd.DataFrame) -> int:
    count = 0
    for low, high in (("L80", "U80"), ("L95", "U95")):
        if {low, high}.issubset(frame.columns):
            count += int((frame[low] > frame[high]).sum())
    return count


def load_and_audit_models(
    config: RunConfig,
) -> Tuple[
    Dict[str, Dict[str, pd.DataFrame]],
    pd.DataFrame,
    pd.DataFrame,
    List[Dict[str, str]],
    List[Dict[str, str]],
]:
    input_dir = Path(config.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    start = pd.Timestamp(config.start) if config.start else None
    end = pd.Timestamp(config.end) if config.end else None
    if start is not None and end is not None and start > end:
        raise ValueError("start must not be later than end")

    candidates, skipped = _candidate_workbooks(
        input_dir,
        include_baselines=config.include_baselines,
    )
    if config.max_models is not None:
        candidates = candidates[: int(config.max_models)]
    if len(candidates) < 2:
        raise ValueError(f"At least two paired workbooks are required; found {len(candidates)}")

    raw_by_model: Dict[str, Dict[str, pd.DataFrame]] = {}
    model_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    used_ids: set[str] = set()
    for path in candidates:
        model_id = _clean_id(path.stem)
        if model_id in used_ids:
            raise ValueError(f"Duplicate model identifier after sanitizing: {model_id}")
        used_ids.add(model_id)
        xl = pd.ExcelFile(path)
        frames: Dict[str, pd.DataFrame] = {}
        for target in TARGETS:
            frame = xl.parse(TARGET_SHEETS[target])
            frame = _normalize_prediction_frame(
                frame,
                source=path,
                target=target,
                start=start,
                end=end,
            )
            stored_mase = _stored_daily_mase(
                xl,
                source=path,
                target=target,
                start=start,
                end=end,
            )
            if stored_mase is not None:
                frame["_daily_MASE"] = frame["date"].map(stored_mase)
                if frame["_daily_MASE"].isna().any():
                    missing_dates = frame.loc[frame["_daily_MASE"].isna(), "date"].drop_duplicates()
                    raise ValueError(
                        f"{path.name}/{target}: stored MASE is missing for "
                        f"{missing_dates.dt.strftime('%Y-%m-%d').tolist()[:5]}"
                    )
                mase_source = TARGET_DAILY_SHEETS[target]
            else:
                frame["_daily_MASE"] = np.nan
                mase_source = "unavailable"
            frames[target] = frame
            daily_counts = frame.groupby("date").size()
            audit_rows.append(
                {
                    "model_id": model_id,
                    "target": target,
                    "rows": len(frame),
                    "days": int(daily_counts.size),
                    "complete_days": int((daily_counts == config.expected_hours).sum()),
                    "min_hours_per_day": int(daily_counts.min()),
                    "max_hours_per_day": int(daily_counts.max()),
                    "start": frame["ts"].min(),
                    "end": frame["ts"].max(),
                    "quantile_crossing_rows": _count_quantile_crossings(frame),
                    "interval_inversions": _count_interval_inversions(frame),
                    "mase_source": mase_source,
                }
            )
        raw_by_model[model_id] = frames
        model_rows.append(
            {
                "model_id": model_id,
                "file": str(path.resolve()),
                "sha256": _file_sha256(path),
                "is_baseline": model_id.lower().startswith("seasonal_naive"),
            }
        )

    duplicate_rows: List[Dict[str, str]] = []
    if config.deduplicate:
        signatures: Dict[str, str] = {}
        for model_id in sorted(list(raw_by_model)):
            signature = _prediction_signature(raw_by_model[model_id])
            if signature in signatures:
                kept = signatures[signature]
                duplicate_rows.append(
                    {"removed_model": model_id, "kept_model": kept, "reason": "exact prediction duplicate"}
                )
                del raw_by_model[model_id]
            else:
                signatures[signature] = model_id

    if len(raw_by_model) < 2:
        raise ValueError("Fewer than two distinct model outputs remain after deduplication")

    models_by_target: Dict[str, Dict[str, pd.DataFrame]] = {target: {} for target in TARGETS}
    for target in TARGETS:
        ids = sorted(raw_by_model)
        if config.alignment == "strict":
            reference = raw_by_model[ids[0]][target]
            reference_ts = pd.DatetimeIndex(reference["ts"])
            reference_y = reference["y_true"].to_numpy(float)
            for model_id in ids:
                frame = raw_by_model[model_id][target]
                current_ts = pd.DatetimeIndex(frame["ts"])
                if not current_ts.equals(reference_ts):
                    raise ValueError(
                        f"{target}: timestamps for {model_id} differ from {ids[0]}; "
                        "use --alignment intersection only if pairwise truncation is intentional"
                    )
                if not np.allclose(frame["y_true"].to_numpy(float), reference_y, equal_nan=False):
                    raise ValueError(f"{target}: actual values differ for {model_id} and {ids[0]}")
                models_by_target[target][model_id] = frame.copy()
        else:
            common = None
            for model_id in ids:
                index = pd.DatetimeIndex(raw_by_model[model_id][target]["ts"])
                common = index if common is None else common.intersection(index)
            if common is None or common.empty:
                raise ValueError(f"{target}: no common timestamps")
            common = common.sort_values()
            reference_y: Optional[np.ndarray] = None
            for model_id in ids:
                frame = raw_by_model[model_id][target].set_index("ts").loc[common].reset_index()
                values = frame["y_true"].to_numpy(float)
                if reference_y is None:
                    reference_y = values
                elif not np.allclose(values, reference_y, equal_nan=False):
                    raise ValueError(f"{target}: actual values differ on the common sample")
                frame["date"] = frame["ts"].dt.normalize()
                frame["horizon"] = frame.groupby("date").cumcount() + 1
                models_by_target[target][model_id] = frame

        day_sets: List[set[pd.Timestamp]] = []
        for model_id, frame in models_by_target[target].items():
            counts = frame.groupby("date").size()
            complete = set(counts[counts == config.expected_hours].index)
            day_sets.append(complete)
        common_days = set.intersection(*day_sets)
        if not common_days:
            raise ValueError(f"{target}: no common complete {config.expected_hours}-hour forecast days")
        if config.alignment == "strict":
            all_days = set(models_by_target[target][ids[0]]["date"].unique())
            if common_days != all_days:
                bad = sorted(all_days - common_days)
                raise ValueError(f"{target}: incomplete forecast days in strict mode: {bad[:5]}")
        else:
            for model_id in ids:
                frame = models_by_target[target][model_id]
                models_by_target[target][model_id] = frame[frame["date"].isin(common_days)].reset_index(drop=True)

    active_ids = set(models_by_target[TARGETS[0]])
    models_table = pd.DataFrame([row for row in model_rows if row["model_id"] in active_ids])
    audit_table = pd.DataFrame([row for row in audit_rows if row["model_id"] in active_ids])
    return models_by_target, models_table, audit_table, skipped, duplicate_rows


def target_unique_models(
    models_by_target: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    deduplicate: bool,
) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], pd.DataFrame]:
    """Remove exact duplicates independently within each forecast target."""

    unique: Dict[str, Dict[str, pd.DataFrame]] = {target: {} for target in TARGETS}
    audit_rows: List[Dict[str, Any]] = []
    for target in TARGETS:
        signatures: Dict[str, str] = {}
        for model_id in sorted(models_by_target[target]):
            frame = models_by_target[target][model_id]
            signature = _target_prediction_signature(frame)
            if deduplicate and signature in signatures:
                representative = signatures[signature]
                reference = models_by_target[target][representative]
                columns = _signature_columns(frame)
                pd.testing.assert_frame_equal(
                    reference[columns].reset_index(drop=True),
                    frame[columns].reset_index(drop=True),
                    check_exact=True,
                )
                audit_rows.append(
                    {
                        "target": target,
                        "duplicate_model_removed": model_id,
                        "representative_model": representative,
                        "rows": len(frame),
                        "prediction_signature_sha256": signature,
                        "reason": "exact target-specific prediction duplicate",
                    }
                )
                continue
            signatures[signature] = model_id
            unique[target][model_id] = frame
        if len(unique[target]) < 2:
            raise ValueError(f"{target}: fewer than two distinct forecast vectors remain")
    return unique, pd.DataFrame(audit_rows)


def _pinball_from_quantiles(frame: pd.DataFrame) -> pd.Series:
    parsed = _quantile_columns(frame)
    if not parsed:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    y = frame["y_true"].to_numpy(float)
    losses: List[np.ndarray] = []
    for tau, column in parsed:
        q = frame[column].to_numpy(float)
        difference = y - q
        losses.append((tau - (difference < 0.0).astype(float)) * difference)
    return pd.Series(np.nanmean(np.column_stack(losses), axis=1), index=frame.index)


def _crps_from_quantiles(frame: pd.DataFrame) -> pd.Series:
    parsed = _quantile_columns(frame)
    if len(parsed) < 2:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    y = frame["y_true"].to_numpy(float)
    taus = np.array([tau for tau, _ in parsed], dtype=float)
    values: List[np.ndarray] = []
    for tau, column in parsed:
        q = frame[column].to_numpy(float)
        difference = y - q
        values.append((tau - (difference < 0.0).astype(float)) * difference)
    integral = np.trapezoid(np.column_stack(values), x=taus, axis=1)
    return pd.Series(2.0 * integral, index=frame.index)


def add_hourly_losses(frame: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    data = frame.copy()
    error = data["y_true"].to_numpy(float) - data["y_pred"].to_numpy(float)
    data["SE"] = error**2
    data["AE"] = np.abs(error)

    pinball_columns = [column for _, column in _pinball_columns(data)]
    if pinball_columns:
        data["PB"] = data[pinball_columns].mean(axis=1)
        pinball_source = "stored pinball columns"
    else:
        data["PB"] = _pinball_from_quantiles(data)
        pinball_source = "reconstructed from quantiles"

    if "CRPS" in data.columns and data["CRPS"].notna().any():
        data["CRPS"] = _numeric(data["CRPS"])
        crps_source = "stored CRPS column"
    else:
        data["CRPS"] = _crps_from_quantiles(data)
        crps_source = "trapezoidal quantile approximation over available levels"

    return data, {"CRPS": crps_source, "PB": pinball_source}


def _interval_summary(frame: pd.DataFrame, coverage: int) -> Dict[str, float]:
    low, high = f"L{coverage:02d}", f"U{coverage:02d}"
    if not {low, high}.issubset(frame.columns):
        return {
            f"Coverage{coverage:02d}": np.nan,
            f"Width{coverage:02d}": np.nan,
            f"IntervalScore{coverage:02d}": np.nan,
        }
    y = frame["y_true"].to_numpy(float)
    lower = frame[low].to_numpy(float)
    upper = frame[high].to_numpy(float)
    nominal = coverage / 100.0
    alpha = 1.0 - nominal
    inside = (y >= lower) & (y <= upper)
    score = (upper - lower).copy()
    score += (2.0 / alpha) * (lower - y) * (y < lower)
    score += (2.0 / alpha) * (y - upper) * (y > upper)
    return {
        f"Coverage{coverage:02d}": float(np.mean(inside)),
        f"Width{coverage:02d}": float(np.mean(upper - lower)),
        f"IntervalScore{coverage:02d}": float(np.mean(score)),
    }


def _overall_metrics(frame: pd.DataFrame) -> Dict[str, float]:
    y = frame["y_true"].to_numpy(float)
    pred = frame["y_pred"].to_numpy(float)
    error = y - pred
    ae = np.abs(error)
    denominator = np.abs(y)
    smape_denominator = np.abs(y) + np.abs(pred)
    row: Dict[str, float] = {
        "n_hours": int(len(frame)),
        "n_days": int(frame["date"].nunique()),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(ae)),
        "MAPE%": float(100.0 * np.mean(ae[denominator > 1e-8] / denominator[denominator > 1e-8])),
        "SMAPE%": float(100.0 * np.mean(2.0 * ae / np.maximum(smape_denominator, 1e-8))),
        "WMAPE%": float(100.0 * np.sum(ae) / max(np.sum(denominator), 1e-8)),
        "Bias_y_minus_pred": float(np.mean(error)),
        "CRPS": float(np.nanmean(frame["CRPS"])),
        "Pinball_mean": float(np.nanmean(frame["PB"])),
        "MASE": (
            float(frame.groupby("date")["_daily_MASE"].first().mean())
            if "_daily_MASE" in frame and frame["_daily_MASE"].notna().all()
            else np.nan
        ),
    }
    row.update(_interval_summary(frame, 80))
    row.update(_interval_summary(frame, 95))
    return row


def build_loss_tables(
    models_by_target: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    regime_split: Optional[str],
) -> Tuple[
    Dict[str, Dict[str, pd.DataFrame]],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    List[Dict[str, str]],
]:
    enriched: Dict[str, Dict[str, pd.DataFrame]] = {target: {} for target in TARGETS}
    daily_rows: List[Dict[str, Any]] = []
    overall_rows: List[Dict[str, Any]] = []
    regime_rows: List[Dict[str, Any]] = []
    source_rows: List[Dict[str, str]] = []
    split = pd.Timestamp(regime_split) if regime_split else None

    for target, models in models_by_target.items():
        for model_id, frame in models.items():
            data, sources = add_hourly_losses(frame)
            for loss in ("CRPS", "PB"):
                if not np.isfinite(data[loss].to_numpy(float)).all():
                    raise ValueError(f"{model_id}/{target}: non-finite {loss} values")
            enriched[target][model_id] = data
            source_rows.append(
                {
                    "model_id": model_id,
                    "target": target,
                    "crps_source": sources["CRPS"],
                    "pinball_source": sources["PB"],
                }
            )

            overall = _overall_metrics(data)
            overall.update(
                {
                    "model_id": model_id,
                    "target": target,
                    "start": data["ts"].min(),
                    "end": data["ts"].max(),
                }
            )
            overall_rows.append(overall)

            for date, day in data.groupby("date", sort=True):
                metrics = _overall_metrics(day)
                daily_rows.append(
                    {
                        "model_id": model_id,
                        "target": target,
                        "date": date,
                        "SE": float(day["SE"].mean()),
                        "AE": float(day["AE"].mean()),
                        "CRPS": float(day["CRPS"].mean()),
                        "PB": float(day["PB"].mean()),
                        "RMSE": metrics["RMSE"],
                        "MAE": metrics["MAE"],
                        "MAPE%": metrics["MAPE%"],
                        "SMAPE%": metrics["SMAPE%"],
                        "WMAPE%": metrics["WMAPE%"],
                        "MASE": metrics["MASE"],
                        "Bias_y_minus_pred": metrics["Bias_y_minus_pred"],
                        "Coverage80": metrics["Coverage80"],
                        "Coverage95": metrics["Coverage95"],
                    }
                )

            if split is not None:
                regimes = {
                    f"before_{split.date()}": data[data["ts"] < split],
                    f"from_{split.date()}": data[data["ts"] >= split],
                }
                for label, subset in regimes.items():
                    if subset.empty:
                        continue
                    metrics = _overall_metrics(subset)
                    metrics.update({"model_id": model_id, "target": target, "regime": label})
                    regime_rows.append(metrics)

    return (
        enriched,
        pd.DataFrame(overall_rows),
        pd.DataFrame(daily_rows),
        pd.DataFrame(regime_rows),
        source_rows,
    )


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
    hac_lag: str,
    min_days: int,
    alpha: float,
    forecast_horizon: int = 1,
) -> DMResult:
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    differential = a[mask] - b[mask]
    n = int(differential.size)
    if n < min_days:
        return DMResult(
            np.nan,
            np.nan,
            n,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
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
        return DMResult(np.nan, np.nan, n, mean, np.nan, np.nan, np.nan, lag, "invalid HAC variance")

    standard_error = math.sqrt(long_run_variance / n)
    h = max(1, int(forecast_horizon))
    hln_term = (n + 1 - 2 * h + h * (h - 1) / n) / n
    if hln_term <= 0:
        return DMResult(np.nan, np.nan, n, mean, standard_error, np.nan, np.nan, lag, "invalid HLN factor")
    hln = math.sqrt(hln_term)
    effective_standard_error = standard_error / hln
    statistic = mean / effective_standard_error
    if student_t is not None:
        p_value = float(2.0 * student_t.sf(abs(statistic), df=n - 1))
        critical = float(student_t.ppf(1.0 - alpha / 2.0, df=n - 1))
    else:
        p_value = float(math.erfc(abs(statistic) / math.sqrt(2.0)))
        critical = 1.959963984540054
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


def _holm_adjust(values: Sequence[float]) -> np.ndarray:
    p_values = np.asarray(values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(p_values))
    if finite.size == 0:
        return adjusted
    ordered = finite[np.argsort(p_values[finite])]
    running = 0.0
    count = len(ordered)
    for rank, index in enumerate(ordered):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def build_daily_dm(
    daily: pd.DataFrame,
    *,
    hac_lag: str,
    min_days: int,
    alpha: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for target in TARGETS:
        target_data = daily[daily["target"] == target]
        ids = sorted(target_data["model_id"].unique())
        for loss in LOSS_CODES:
            block: List[Dict[str, Any]] = []
            for model_a, model_b in itertools.combinations(ids, 2):
                a = target_data[target_data["model_id"] == model_a][["date", loss]].rename(columns={loss: "a"})
                b = target_data[target_data["model_id"] == model_b][["date", loss]].rename(columns={loss: "b"})
                merged = a.merge(b, on="date", how="inner", validate="one_to_one")
                result = dm_test_losses(
                    merged["a"],
                    merged["b"],
                    hac_lag=hac_lag,
                    min_days=min_days,
                    alpha=alpha,
                    forecast_horizon=1,
                )
                row = asdict(result)
                row.update({"target": target, "loss": loss, "model_a": model_a, "model_b": model_b})
                block.append(row)
            adjusted = _holm_adjust([row["p_value"] for row in block])
            for row, p_holm in zip(block, adjusted):
                row["p_holm"] = float(p_holm) if np.isfinite(p_holm) else np.nan
                if np.isfinite(p_holm) and p_holm < alpha:
                    row["better_model"] = row["model_a"] if row["mean_difference"] < 0 else row["model_b"]
                else:
                    row["better_model"] = ""
                rows.append(row)
    return pd.DataFrame(rows)


def build_horizon_dm(
    enriched: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    hac_lag: str,
    min_days: int,
    alpha: float,
    expected_hours: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for target, models in enriched.items():
        ids = sorted(models)
        for loss in LOSS_CODES:
            target_loss_rows: List[Dict[str, Any]] = []
            for horizon in range(1, expected_hours + 1):
                horizon_rows: List[Dict[str, Any]] = []
                for model_a, model_b in itertools.combinations(ids, 2):
                    a = models[model_a]
                    b = models[model_b]
                    aa = a[a["horizon"] == horizon][["date", loss]].rename(columns={loss: "a"})
                    bb = b[b["horizon"] == horizon][["date", loss]].rename(columns={loss: "b"})
                    merged = aa.merge(bb, on="date", how="inner", validate="one_to_one")
                    result = dm_test_losses(
                        merged["a"],
                        merged["b"],
                        hac_lag=hac_lag,
                        min_days=min_days,
                        alpha=alpha,
                        forecast_horizon=1,
                    )
                    row = asdict(result)
                    row.update(
                        {
                            "target": target,
                            "loss": loss,
                            "horizon": horizon,
                            "model_a": model_a,
                            "model_b": model_b,
                        }
                    )
                    horizon_rows.append(row)
                within = _holm_adjust([row["p_value"] for row in horizon_rows])
                for row, adjusted in zip(horizon_rows, within):
                    row["p_holm_within_horizon"] = float(adjusted) if np.isfinite(adjusted) else np.nan
                target_loss_rows.extend(horizon_rows)
            global_adjusted = _holm_adjust([row["p_value"] for row in target_loss_rows])
            for row, adjusted in zip(target_loss_rows, global_adjusted):
                row["p_holm_global"] = float(adjusted) if np.isfinite(adjusted) else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def build_league(dm_pairs: pd.DataFrame, *, alpha: float) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (target, loss), group in dm_pairs.groupby(["target", "loss"], sort=True):
        ids = sorted(set(group["model_a"]) | set(group["model_b"]))
        wins = {model_id: 0 for model_id in ids}
        losses = {model_id: 0 for model_id in ids}
        for row in group.itertuples(index=False):
            if not np.isfinite(row.p_holm) or row.p_holm >= alpha:
                continue
            better = row.model_a if row.mean_difference < 0 else row.model_b
            worse = row.model_b if better == row.model_a else row.model_a
            wins[better] += 1
            losses[worse] += 1
        for model_id in ids:
            rows.append(
                {
                    "target": target,
                    "loss": loss,
                    "model_id": model_id,
                    "significant_wins": wins[model_id],
                    "significant_losses": losses[model_id],
                    "net_wins": wins[model_id] - losses[model_id],
                    "non_significant_pairs": len(ids) - 1 - wins[model_id] - losses[model_id],
                }
            )
    return pd.DataFrame(rows)


def _matrix_from_pairs(
    group: pd.DataFrame,
    model_ids: Sequence[str],
    value_column: str,
    *,
    antisymmetric: bool,
    diagonal: float,
) -> pd.DataFrame:
    matrix = pd.DataFrame(np.nan, index=model_ids, columns=model_ids, dtype=float)
    for model_id in model_ids:
        matrix.loc[model_id, model_id] = diagonal
    for row in group.itertuples(index=False):
        value = float(getattr(row, value_column))
        matrix.loc[row.model_a, row.model_b] = value
        matrix.loc[row.model_b, row.model_a] = -value if antisymmetric else value
    return matrix


def build_dm_matrices(dm_pairs: pd.DataFrame) -> Dict[Tuple[str, str, str], pd.DataFrame]:
    matrices: Dict[Tuple[str, str, str], pd.DataFrame] = {}
    for (target, loss), group in dm_pairs.groupby(["target", "loss"], sort=True):
        ids = sorted(set(group["model_a"]) | set(group["model_b"]))
        specifications = (
            ("stat", "statistic", True, 0.0),
            ("p", "p_value", False, 1.0),
            ("holm", "p_holm", False, 1.0),
            ("diff", "mean_difference", True, 0.0),
            ("n", "n", False, float(group["n"].max())),
        )
        for suffix, column, antisymmetric, diagonal in specifications:
            matrices[(target, loss, suffix)] = _matrix_from_pairs(
                group,
                ids,
                column,
                antisymmetric=antisymmetric,
                diagonal=diagonal,
            )
    return matrices


def model_short_label(model_id: str) -> str:
    """Return the compact system names used in tables and article figures."""

    lower = model_id.lower()
    if lower.startswith("seasonal_naive_lag24"):
        return "SNaive-24"
    if lower.startswith("seasonal_naive_lag168"):
        return "SNaive-168"
    if lower.startswith("lagdrop_"):
        return "No-history"
    if lower.startswith("mi_top_"):
        return "MI-9"
    if lower.startswith("sfs_shared_"):
        return "SFS-shared"

    lag_label = "L1"
    suffix = re.search(r"_(0|1|2)$", model_id)
    if suffix:
        lag_label = {"0": "L1", "1": "L24", "2": "L168"}[suffix.group(1)]
    if lower.startswith("mutual_lags_"):
        return f"MUT-{lag_label}"
    if lower.startswith("own_lags_"):
        return f"OWN-{lag_label}"
    if lower.startswith("sfs_"):
        return f"SFS-{lag_label}"
    return model_id


def build_complete_system_selection(
    overall: pd.DataFrame,
    *,
    selected_system: Optional[str],
) -> Tuple[pd.DataFrame, str]:
    """Rank complete P-Q systems by mean target-normalized RMSE."""

    rmse = overall.pivot(index="model_id", columns="target", values="RMSE")
    rmse = rmse.dropna(subset=list(TARGETS))
    if rmse.empty:
        raise ValueError("No complete P-Q system is available for joint selection")
    minima = rmse.min(axis=0)
    table = pd.DataFrame(
        {
            "model_id": rmse.index,
            "short_label": [model_short_label(model_id) for model_id in rmse.index],
            "P_RMSE": rmse["P_Power"].to_numpy(float),
            "Q_RMSE": rmse["Q_Power"].to_numpy(float),
            "P_normalized_RMSE": (rmse["P_Power"] / minima["P_Power"]).to_numpy(float),
            "Q_normalized_RMSE": (rmse["Q_Power"] / minima["Q_Power"]).to_numpy(float),
        }
    )
    table["mean_target_normalized_RMSE"] = table[
        ["P_normalized_RMSE", "Q_normalized_RMSE"]
    ].mean(axis=1)
    table = table.sort_values(
        ["mean_target_normalized_RMSE", "P_RMSE", "Q_RMSE", "model_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    retained = selected_system or str(table.iloc[0]["model_id"])
    if retained not in set(table["model_id"]):
        raise ValueError(
            f"Selected complete system {retained!r} is absent from one or both targets"
        )
    table["selected_complete_system"] = table["model_id"].eq(retained)
    return table, retained


def build_selected_contrasts(
    dm_pairs: pd.DataFrame,
    *,
    selected_system: str,
) -> pd.DataFrame:
    """Orient selected-versus-all SE and AE contrasts as selected minus comparator."""

    rows: List[Dict[str, Any]] = []
    for target in TARGETS:
        for loss in ("SE", "AE"):
            family = dm_pairs[
                (dm_pairs["target"] == target) & (dm_pairs["loss"] == loss)
            ]
            for row in family.itertuples(index=False):
                if selected_system not in {row.model_a, row.model_b}:
                    continue
                selected_is_a = row.model_a == selected_system
                comparator = row.model_b if selected_is_a else row.model_a
                sign = 1.0 if selected_is_a else -1.0
                ci_low = row.ci_low if selected_is_a else -row.ci_high
                ci_high = row.ci_high if selected_is_a else -row.ci_low
                rows.append(
                    {
                        "target": target,
                        "loss": loss,
                        "selected_model": selected_system,
                        "selected_label": model_short_label(selected_system),
                        "comparator_model": comparator,
                        "comparator_label": model_short_label(comparator),
                        "n": row.n,
                        "mean_difference_selected_minus_comparator": sign
                        * row.mean_difference,
                        "ci_low_pointwise_95": ci_low,
                        "ci_high_pointwise_95": ci_high,
                        "dm_statistic": sign * row.statistic,
                        "raw_p": row.p_value,
                        "holm_p": row.p_holm,
                        "significant_0.05": bool(
                            np.isfinite(row.p_holm) and row.p_holm < 0.05
                        ),
                    }
                )
    return pd.DataFrame(rows)


_EFFECT_STOPS = (
    (-80.0, "#0072b2"),
    (-20.0, "#cfe8f3"),
    (0.0, "#f7f7f7"),
    (20.0, "#f7d9c4"),
    (80.0, "#d55e00"),
)
_DIAGONAL_COLOR = "#565a5e"
_SIGNIFICANT_BORDER = "#202326"
_RETAINED_BORDER = "#7a3e9d"
_MARGINAL_COLOR = "#0072b2"
_TEXT_COLOR = "#202326"


def _hex_rgb(value: str) -> Tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def _interpolate_effect_color(value: float) -> Tuple[int, int, int]:
    clipped = min(80.0, max(-80.0, value))
    for (left_value, left_color), (right_value, right_color) in zip(
        _EFFECT_STOPS[:-1], _EFFECT_STOPS[1:]
    ):
        if left_value <= clipped <= right_value:
            fraction = (clipped - left_value) / (right_value - left_value)
            left = _hex_rgb(left_color)
            right = _hex_rgb(right_color)
            return tuple(
                round(left[channel] + fraction * (right[channel] - left[channel]))
                for channel in range(3)
            )
    return _hex_rgb(_EFFECT_STOPS[-1][1])


def _luminance(color: Tuple[int, int, int]) -> float:
    red, green, blue = (channel / 255.0 for channel in color)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _publication_font(size: int, *, bold: bool) -> Any:
    if ImageFont is None:
        raise RuntimeError("Pillow is required for the publication matrix")
    candidates = [
        Path(r"C:\Windows\Fonts\timesbd.ttf" if bold else r"C:\Windows\Fonts\times.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf")
        if bold
        else Path("/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf")
        if bold
        else Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    for name in (
        "DejaVuSerif-Bold.ttf" if bold else "DejaVuSerif.ttf",
        "LiberationSerif-Bold.ttf" if bold else "LiberationSerif-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    raise RuntimeError("No publication-quality serif font could be located")


def _format_adjusted_p(value: float) -> str:
    if value < 0.001:
        return "<.001"
    if value < 1.0:
        return f"{value:.3f}".removeprefix("0")
    return "1.000"


def _target_figure_label(
    model_id: str,
    *,
    target: str,
    target_duplicates: pd.DataFrame,
) -> str:
    label = model_short_label(model_id)
    if not target_duplicates.empty:
        duplicate = target_duplicates[
            (target_duplicates["target"] == target)
            & (target_duplicates["representative_model"] == model_id)
        ]
        if not duplicate.empty and label == "SFS-L1":
            return "SFS-L1/shared"
    return label


def _publication_panel_data(
    dm_pairs: pd.DataFrame,
    overall: pd.DataFrame,
    *,
    target: str,
    target_duplicates: pd.DataFrame,
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    family = dm_pairs[
        (dm_pairs["target"] == target) & (dm_pairs["loss"] == "SE")
    ].copy()
    model_ids = sorted(set(family["model_a"]) | set(family["model_b"]))
    expected = len(model_ids) * (len(model_ids) - 1) // 2
    if len(family) != expected:
        raise ValueError(
            f"{target}/SE: expected {expected} unique pairs, found {len(family)}"
        )
    rmse_lookup = (
        overall[overall["target"] == target]
        .drop_duplicates("model_id")
        .set_index("model_id")["RMSE"]
        .to_dict()
    )
    missing = [model_id for model_id in model_ids if model_id not in rmse_lookup]
    if missing:
        raise ValueError(f"{target}: RMSE is unavailable for {missing}")
    ordered = sorted(model_ids, key=lambda model_id: (rmse_lookup[model_id], model_id))
    labels = [
        _target_figure_label(
            model_id,
            target=target,
            target_duplicates=target_duplicates,
        )
        for model_id in ordered
    ]
    positions = {model_id: index for index, model_id in enumerate(ordered)}
    count = len(ordered)
    p_values = np.full((count, count), np.nan, dtype=float)
    significant = np.zeros((count, count), dtype=bool)
    for row in family.itertuples(index=False):
        left = positions[row.model_a]
        right = positions[row.model_b]
        p_values[left, right] = p_values[right, left] = float(row.p_holm)
        significant[left, right] = significant[right, left] = bool(
            np.isfinite(row.p_holm) and row.p_holm < 0.05
        )

    rmse_values = np.array([rmse_lookup[model_id] for model_id in ordered], dtype=float)
    effects = 200.0 * (
        rmse_values[:, np.newaxis] - rmse_values[np.newaxis, :]
    ) / (rmse_values[:, np.newaxis] + rmse_values[np.newaxis, :])
    np.fill_diagonal(effects, np.nan)

    rows: List[Dict[str, Any]] = []
    for row_index, row_model in enumerate(ordered):
        for column_index, column_model in enumerate(ordered):
            if row_index == column_index:
                continue
            rows.append(
                {
                    "target": target,
                    "row_rank": row_index + 1,
                    "row_model_id": row_model,
                    "row_model": labels[row_index],
                    "column_rank": column_index + 1,
                    "column_model_id": column_model,
                    "column_model": labels[column_index],
                    "row_RMSE": rmse_values[row_index],
                    "column_RMSE": rmse_values[column_index],
                    "symmetric_RMSE_difference_percent": effects[
                        row_index, column_index
                    ],
                    "Holm_p_unique": p_values[row_index, column_index],
                    "significant_0.05": significant[row_index, column_index],
                    "direction": (
                        "row_lower"
                        if effects[row_index, column_index] < 0
                        else "row_higher"
                    ),
                }
            )
    return ordered, labels, p_values, significant, effects, rows


def _text_size(draw: Any, text: str, font: Any) -> Tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def _draw_centered_text(
    draw: Any,
    box: Tuple[float, float, float, float],
    text: str,
    font: Any,
    fill: Any,
) -> None:
    left, top, right, bottom = box
    width, height = _text_size(draw, text, font)
    draw.text(
        (
            left + (right - left - width) / 2,
            top + (bottom - top - height) / 2 - 2,
        ),
        text,
        font=font,
        fill=fill,
    )


def _draw_publication_panel(
    draw: Any,
    *,
    panel_start: int,
    title: str,
    ordered: Sequence[str],
    labels: Sequence[str],
    p_values: np.ndarray,
    significant: np.ndarray,
    effects: np.ndarray,
    retained_system: str,
    matrix_top: int,
    matrix_offset: int,
    cell_size: int,
    fonts: Mapping[str, Any],
) -> None:
    matrix_left = panel_start + matrix_offset
    label_right = matrix_left - 28
    draw.text(
        (panel_start + 45, 52),
        title,
        font=fonts["title"],
        fill=_TEXT_COLOR,
    )
    for row_index in range(len(ordered)):
        for column_index in range(len(ordered)):
            left = matrix_left + column_index * cell_size
            top = matrix_top + row_index * cell_size
            box = (left, top, left + cell_size, top + cell_size)
            if row_index == column_index:
                text = "-"
                background = _hex_rgb(_DIAGONAL_COLOR)
                text_color: Any = "white"
                font = fonts["cell_bold"]
            else:
                text = _format_adjusted_p(float(p_values[row_index, column_index]))
                background = _interpolate_effect_color(
                    float(effects[row_index, column_index])
                )
                text_color = "white" if _luminance(background) < 0.52 else "#34383b"
                font = (
                    fonts["cell_bold"]
                    if significant[row_index, column_index]
                    else fonts["cell"]
                )
            draw.rectangle(box, fill=background, outline="white", width=4)
            if row_index != column_index and significant[row_index, column_index]:
                draw.rectangle(
                    (left + 2, top + 2, left + cell_size - 2, top + cell_size - 2),
                    outline=_SIGNIFICANT_BORDER,
                    width=4,
                )
            _draw_centered_text(draw, box, text, font, text_color)

    for index, (model_id, label) in enumerate(zip(ordered, labels)):
        label_text = f"{index + 1}. {label}"
        label_font = fonts["label"]
        rank_font = fonts["rank"]
        label_color = _TEXT_COLOR
        if index == 0 and model_id != retained_system:
            label_font = fonts["label_bold"]
            rank_font = fonts["rank_bold"]
            label_color = _MARGINAL_COLOR
        if model_id == retained_system:
            label_font = fonts["label_bold"]
            rank_font = fonts["rank_bold"]
            label_color = _RETAINED_BORDER
        rank_box = (
            matrix_left + index * cell_size,
            matrix_top - 82,
            matrix_left + (index + 1) * cell_size,
            matrix_top - 10,
        )
        _draw_centered_text(
            draw,
            rank_box,
            str(index + 1),
            rank_font,
            label_color,
        )
        label_width, label_height = _text_size(draw, label_text, label_font)
        label_y = matrix_top + index * cell_size + (cell_size - label_height) / 2 - 2
        draw.text(
            (label_right - label_width, label_y),
            label_text,
            font=label_font,
            fill=label_color,
        )

    if retained_system in ordered:
        retained_index = ordered.index(retained_system)
        row_top = matrix_top + retained_index * cell_size
        draw.rectangle(
            (
                matrix_left - 5,
                row_top - 5,
                matrix_left + len(ordered) * cell_size + 5,
                row_top + cell_size + 5,
            ),
            outline=_RETAINED_BORDER,
            width=8,
        )


def _save_publication_pdf(image: Any, path: Path, *, dpi: int) -> None:
    if canvas is not None and ImageReader is not None:
        page_width = image.width / dpi * 72.0
        page_height = image.height / dpi * 72.0
        pdf = canvas.Canvas(str(path), pagesize=(page_width, page_height))
        pdf.drawImage(
            ImageReader(image),
            0,
            0,
            width=page_width,
            height=page_height,
            preserveAspectRatio=True,
            mask="auto",
        )
        pdf.showPage()
        pdf.save()
    else:
        image.save(path, format="PDF", resolution=float(dpi))


def create_publication_outputs(
    run_dir: Path,
    dm_pairs: pd.DataFrame,
    overall: pd.DataFrame,
    target_duplicates: pd.DataFrame,
    selection: pd.DataFrame,
    *,
    retained_system: str,
    dpi: int,
) -> Dict[str, Any]:
    """Build the target-unique all-versus-all matrix and its numerical audit."""

    if Image is None or ImageDraw is None or ImageFont is None:
        raise RuntimeError(
            "Pillow is required for the publication matrix; install package 'Pillow'"
        )
    publication_dir = run_dir / "publication"
    publication_dir.mkdir(parents=True, exist_ok=True)
    panel_data: Dict[str, Tuple[Any, ...]] = {}
    matrix_rows: List[Dict[str, Any]] = []
    for target in TARGETS:
        panel_data[target] = _publication_panel_data(
            dm_pairs,
            overall,
            target=target,
            target_duplicates=target_duplicates,
        )
        matrix_rows.extend(panel_data[target][-1])

    max_models = max(len(panel_data[target][0]) for target in TARGETS)
    cell_size = min(88, max(58, 1232 // max_models))
    matrix_top = 300
    canvas_width = 4320
    canvas_height = max(900, matrix_top + max_models * cell_size + 128)
    fonts = {
        "title": _publication_font(69, bold=True),
        "rank": _publication_font(50, bold=False),
        "rank_bold": _publication_font(50, bold=True),
        "label": _publication_font(49, bold=False),
        "label_bold": _publication_font(49, bold=True),
        "cell": _publication_font(36, bold=False),
        "cell_bold": _publication_font(36, bold=True),
    }
    image = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(image)
    for panel_index, target in enumerate(TARGETS):
        ordered, labels, p_values, significant, effects, _ = panel_data[target]
        _draw_publication_panel(
            draw,
            panel_start=panel_index * (canvas_width // 2),
            title=f"({chr(97 + panel_index)}) {TARGET_NAME[target]}",
            ordered=ordered,
            labels=labels,
            p_values=p_values,
            significant=significant,
            effects=effects,
            retained_system=retained_system,
            matrix_top=matrix_top,
            matrix_offset=850 if panel_index == 0 else 990,
            cell_size=cell_size,
            fonts=fonts,
        )

    png_path = publication_dir / "dm_pairwise_unique_q_full_width.png"
    pdf_path = publication_dir / "dm_pairwise_unique_q_full_width.pdf"
    values_path = publication_dir / "dm_pairwise_unique_q_values.csv"
    contrasts_path = publication_dir / "dm_selected_contrasts_unique_q.csv"
    selection_path = publication_dir / "complete_system_selection.csv"
    corrected_pairs_path = publication_dir / "dm_pairs_daily_target_unique.csv"
    duplicate_audit_path = (
        publication_dir / "dm_target_specific_duplicate_audit.csv"
    )
    image.save(png_path, format="PNG", dpi=(dpi, dpi), optimize=True)
    _save_publication_pdf(image, pdf_path, dpi=dpi)
    pd.DataFrame(matrix_rows).to_csv(values_path, index=False)
    build_selected_contrasts(
        dm_pairs,
        selected_system=retained_system,
    ).to_csv(contrasts_path, index=False)
    selection.to_csv(selection_path, index=False)
    corrected_pairs = dm_pairs.copy()
    corrected_pairs["p_holm_unique"] = corrected_pairs["p_holm"]
    corrected_pairs["family_size_unique"] = corrected_pairs.groupby(
        ["target", "loss"]
    )["model_a"].transform("size")
    corrected_pairs.to_csv(corrected_pairs_path, index=False)
    target_duplicates.to_csv(duplicate_audit_path, index=False)
    outputs = {
        path.name: _file_sha256(path)
        for path in (
            png_path,
            pdf_path,
            values_path,
            contrasts_path,
            selection_path,
            corrected_pairs_path,
            duplicate_audit_path,
        )
    }
    manifest = {
        "retained_complete_system": retained_system,
        "retained_complete_system_label": model_short_label(retained_system),
        "matrix_loss": "daily mean squared error",
        "cell_value": "Holm-adjusted two-sided DM p-value",
        "cell_color": (
            "symmetric RMSE difference, 200*(row-column)/(row+column); "
            "blue means lower row RMSE and orange means higher row RMSE"
        ),
        "dark_cell_border": "Holm-adjusted p < 0.05",
        "purple_row": "retained complete P-Q system",
        "blue_label": "target-specific minimum RMSE when different from retained system",
        "target_model_counts": {
            target: len(panel_data[target][0]) for target in TARGETS
        },
        "target_pair_counts": {
            target: int(
                len(
                    dm_pairs[
                        (dm_pairs["target"] == target)
                        & (dm_pairs["loss"] == "SE")
                    ]
                )
            )
            for target in TARGETS
        },
        "outputs": outputs,
    }
    manifest_path = publication_dir / "dm_pairwise_unique_q_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    outputs[manifest_path.name] = _file_sha256(manifest_path)
    return {
        "directory": str(publication_dir),
        "outputs": outputs,
        "target_model_counts": manifest["target_model_counts"],
        "target_pair_counts": manifest["target_pair_counts"],
    }


def _model_color_map(model_ids: Sequence[str]) -> Dict[str, Any]:
    cmap = plt.get_cmap("tab20")
    return {model_id: cmap(index % 20) for index, model_id in enumerate(sorted(model_ids))}


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, facecolor="white")
    plt.close()


def plot_forecast(frame: pd.DataFrame, model_id: str, target: str, path: Path, metrics: Mapping[str, Any]) -> None:
    plt.figure(figsize=(14, 6))
    if {"L95", "U95"}.issubset(frame.columns):
        plt.fill_between(frame["ts"], frame["L95"], frame["U95"], color="#9ecae1", alpha=0.28, label="95% interval")
    if {"L80", "U80"}.issubset(frame.columns):
        plt.fill_between(frame["ts"], frame["L80"], frame["U80"], color="#4292c6", alpha=0.30, label="80% interval")
    plt.plot(frame["ts"], frame["y_true"], color="#202020", linewidth=1.0, label="Actual")
    plt.plot(frame["ts"], frame["y_pred"], color="#d62728", linewidth=1.0, label="Forecast")
    plt.title(
        f"{TARGET_NAME[target]}: {model_short_label(model_id)} - "
        "recursive 24-hour forecasts"
    )
    plt.xlabel("Time")
    plt.ylabel(TARGET_UNIT[target])
    plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.65)
    plt.legend(loc="best", ncol=2)
    annotation = (
        f"RMSE {metrics['RMSE']:.2f} | MAE {metrics['MAE']:.2f} | "
        f"C80 {metrics['Coverage80']:.1%} | C95 {metrics['Coverage95']:.1%}"
    )
    plt.figtext(0.5, 0.01, annotation, ha="center", fontsize=9)
    plt.tight_layout(rect=(0, 0.035, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, facecolor="white")
    plt.close()


def plot_daily_probabilistic_scores(daily: pd.DataFrame, model_id: str, target: str, path: Path) -> None:
    block = daily[(daily["model_id"] == model_id) & (daily["target"] == target)].sort_values("date")
    plt.figure(figsize=(12, 5))
    plt.plot(block["date"], block["CRPS"], linewidth=1.3, label="Daily mean CRPS")
    plt.plot(block["date"], block["PB"], linewidth=1.3, label="Daily mean pinball")
    plt.title(
        f"{TARGET_NAME[target]}: {model_short_label(model_id)} - "
        "daily probabilistic scores"
    )
    plt.xlabel("Date")
    plt.ylabel("Loss")
    plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.65)
    plt.legend(loc="best")
    _save_figure(path)


def plot_global_daily(daily: pd.DataFrame, target: str, metric: str, path: Path) -> None:
    block = daily[daily["target"] == target]
    ids = sorted(block["model_id"].unique())
    colors = _model_color_map(ids)
    plt.figure(figsize=(14, 6))
    for model_id in ids:
        current = block[block["model_id"] == model_id].sort_values("date")
        plt.plot(
            current["date"],
            current[metric],
            label=model_short_label(model_id),
            color=colors[model_id],
            linewidth=1.15,
            alpha=0.9,
        )
    plt.title(f"{TARGET_NAME[target]}: daily {metric} across evaluated systems")
    plt.xlabel("Date")
    plt.ylabel(metric)
    plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.65)
    columns = 2 if len(ids) <= 16 else 3
    plt.legend(loc="upper left", ncol=columns, fontsize=7.5, frameon=True)
    _save_figure(path)


def plot_dm_heatmap(
    group: pd.DataFrame,
    target: str,
    loss: str,
    path: Path,
    *,
    alpha: float,
) -> None:
    ids = sorted(set(group["model_a"]) | set(group["model_b"]))
    values = np.zeros((len(ids), len(ids)), dtype=float)
    positions = {model_id: index for index, model_id in enumerate(ids)}
    for row in group.itertuples(index=False):
        i, j = positions[row.model_a], positions[row.model_b]
        if np.isfinite(row.p_holm) and row.p_holm < alpha:
            row_better = row.mean_difference < 0
            values[i, j] = 1.0 if row_better else -1.0
            values[j, i] = -values[i, j]
    np.fill_diagonal(values, 2.0)
    cmap = ListedColormap(["#d73027", "#d9d9d9", "#2ca02c", "#4d4d4d"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], cmap.N)
    size = max(8.0, 0.55 * len(ids) + 4.0)
    plt.figure(figsize=(size + 3.0, size))
    plt.imshow(values, cmap=cmap, norm=norm, aspect="equal")
    labels = [model_short_label(model_id) for model_id in ids]
    plt.xticks(range(len(ids)), labels, rotation=55, ha="right", fontsize=8)
    plt.yticks(range(len(ids)), labels, fontsize=8)
    plt.title(
        f"Daily-trajectory DM: {TARGET_NAME[target]}, "
        f"{LOSS_LABELS[loss]} (Holm alpha={alpha:g})"
    )
    plt.grid(False)
    plt.legend(
        handles=[
            Patch(color="#2ca02c", label="Row significantly better"),
            Patch(color="#d73027", label="Row significantly worse"),
            Patch(color="#d9d9d9", label="No significant difference"),
        ],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
    )
    _save_figure(path)


def plot_league(league: pd.DataFrame, target: str, loss: str, path: Path) -> None:
    block = league[(league["target"] == target) & (league["loss"] == loss)].copy()
    block = block.sort_values(["net_wins", "significant_wins", "model_id"], ascending=[True, True, False])
    colors = ["#2ca02c" if value > 0 else "#d73027" if value < 0 else "#969696" for value in block["net_wins"]]
    plt.figure(figsize=(11, max(5, 0.42 * len(block) + 2)))
    labels = [model_short_label(model_id) for model_id in block["model_id"]]
    bars = plt.barh(labels, block["net_wins"], color=colors)
    for bar, row in zip(bars, block.itertuples(index=False)):
        x = bar.get_width()
        label = f"W {row.significant_wins} / L {row.significant_losses} / NS {row.non_significant_pairs}"
        text_x = x / 2.0 if x != 0 else 0.15
        plt.text(
            text_x,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="center" if x != 0 else "left",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2},
        )
    plt.axvline(0.0, color="#202020", linewidth=0.8)
    plt.title(
        f"Holm-adjusted DM league: {TARGET_NAME[target]}, {LOSS_LABELS[loss]}"
    )
    plt.xlabel("Significant wins minus significant losses")
    plt.ylabel("")
    plt.grid(True, axis="x", linestyle=":", linewidth=0.6, alpha=0.65)
    _save_figure(path)


def create_plots(
    run_dir: Path,
    enriched: Mapping[str, Mapping[str, pd.DataFrame]],
    overall: pd.DataFrame,
    daily: pd.DataFrame,
    dm_pairs: pd.DataFrame,
    league: pd.DataFrame,
    *,
    alpha: float,
    per_model: bool,
) -> None:
    plot_dir = run_dir / "plots"
    if per_model:
        for target, models in enriched.items():
            for model_id, frame in models.items():
                metrics = overall[(overall["target"] == target) & (overall["model_id"] == model_id)].iloc[0]
                target_dir = plot_dir / target
                plot_forecast(frame, model_id, target, target_dir / f"forecast_{model_id}.png", metrics)
                plot_daily_probabilistic_scores(
                    daily,
                    model_id,
                    target,
                    target_dir / f"probabilistic_scores_{model_id}.png",
                )
    for target in TARGETS:
        for metric in (
            "RMSE",
            "MAE",
            "MAPE%",
            "SMAPE%",
            "WMAPE%",
            "MASE",
            "CRPS",
            "PB",
        ):
            if metric not in daily or daily[metric].isna().all():
                continue
            safe = metric.replace("%", "pct")
            plot_global_daily(daily, target, metric, plot_dir / f"global_daily_{safe}_{TARGET_SHORT[target]}.png")
        for loss in LOSS_CODES:
            group = dm_pairs[(dm_pairs["target"] == target) & (dm_pairs["loss"] == loss)]
            plot_dm_heatmap(group, target, loss, plot_dir / f"DM_heatmap_{TARGET_SHORT[target]}_{loss}.png", alpha=alpha)
            plot_league(league, target, loss, plot_dir / f"DM_league_{TARGET_SHORT[target]}_{loss}.png")


def _environment() -> Dict[str, Any]:
    packages: Dict[str, Optional[str]] = {}
    for package in ("numpy", "pandas", "scipy", "matplotlib", "openpyxl", "xlsxwriter"):
        try:
            module = __import__(package)
            packages[package] = getattr(module, "__version__", None)
        except Exception:
            packages[package] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def write_outputs(
    run_dir: Path,
    config: RunConfig,
    models_table: pd.DataFrame,
    audit_table: pd.DataFrame,
    skipped: Sequence[Mapping[str, str]],
    duplicates: Sequence[Mapping[str, str]],
    target_duplicates: pd.DataFrame,
    sources: Sequence[Mapping[str, str]],
    overall: pd.DataFrame,
    daily: pd.DataFrame,
    regimes: pd.DataFrame,
    dm_pairs: pd.DataFrame,
    horizon_dm: pd.DataFrame,
    league: pd.DataFrame,
    matrices: Mapping[Tuple[str, str, str], pd.DataFrame],
    selection: pd.DataFrame,
    retained_system: str,
    publication_info: Mapping[str, Any],
) -> None:
    overall.to_csv(run_dir / "model_metrics.csv", index=False)
    daily.to_csv(run_dir / "daily_losses.csv", index=False)
    dm_pairs.to_csv(run_dir / "dm_pairs_daily.csv", index=False)
    league.to_csv(run_dir / "dm_league.csv", index=False)
    target_duplicates.to_csv(run_dir / "target_specific_duplicates.csv", index=False)
    selection.to_csv(run_dir / "complete_system_selection.csv", index=False)
    if not regimes.empty:
        regimes.to_csv(run_dir / "regime_metrics.csv", index=False)
    if not horizon_dm.empty:
        horizon_dm.to_csv(run_dir / "dm_horizon.csv", index=False)

    protocol = pd.DataFrame(
        [
            ("primary_unit", "one mean loss per complete 24-hour forecast day"),
            ("primary_frequency", "daily forecast origins"),
            ("losses", ", ".join(LOSS_CODES)),
            ("dm_alternative", "two-sided unequal predictive accuracy"),
            ("difference_sign", "negative model_a minus model_b means model_a is better"),
            ("variance", f"Bartlett HAC; lag={config.hac_lag}"),
            ("small_sample", "Harvey-Leybourne-Newbold correction with origin horizon 1"),
            ("multiplicity", "Holm correction within each target and loss family"),
            (
                "duplicate_policy",
                "exact forecast duplicates removed separately within each target before DM testing",
            ),
            ("horizon_tests", "secondary; Holm within horizon and globally across horizons"),
            (
                "MASE",
                "read from evaluator daily-metric sheets; denominator was fixed before evaluation",
            ),
            ("retained_complete_system", retained_system),
        ],
        columns=["item", "value"],
    )
    workbook = run_dir / "DM_across_models.xlsx"
    with pd.ExcelWriter(workbook, engine="xlsxwriter") as writer:
        protocol.to_excel(writer, sheet_name="protocol", index=False)
        models_table.to_excel(writer, sheet_name="models", index=False)
        audit_table.to_excel(writer, sheet_name="data_audit", index=False)
        pd.DataFrame(skipped).to_excel(writer, sheet_name="skipped_files", index=False)
        pd.DataFrame(duplicates).to_excel(writer, sheet_name="paired_duplicates", index=False)
        target_duplicates.to_excel(writer, sheet_name="target_duplicates", index=False)
        pd.DataFrame(sources).to_excel(writer, sheet_name="loss_sources", index=False)
        overall.to_excel(writer, sheet_name="overall_metrics", index=False)
        daily.to_excel(writer, sheet_name="daily_losses", index=False)
        selection.to_excel(writer, sheet_name="system_selection", index=False)
        dm_pairs.to_excel(writer, sheet_name="DM_pairs_daily", index=False)
        league.to_excel(writer, sheet_name="DM_league", index=False)
        if not regimes.empty:
            regimes.to_excel(writer, sheet_name="regime_metrics", index=False)
        if not horizon_dm.empty:
            horizon_dm.to_excel(writer, sheet_name="horizon_DM", index=False)
        for (target, loss, suffix), matrix in matrices.items():
            sheet = f"{TARGET_SHORT[target]}_{loss}_{suffix}"
            matrix.to_excel(writer, sheet_name=sheet[:31])

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "script_sha256": _file_sha256(Path(__file__)),
        "config": asdict(config),
        "input_files": models_table.to_dict(orient="records"),
        "skipped_files": list(skipped),
        "exact_paired_duplicates_removed": list(duplicates),
        "exact_target_specific_duplicates_removed": target_duplicates.to_dict(
            orient="records"
        ),
        "retained_complete_system": retained_system,
        "retained_complete_system_label": model_short_label(retained_system),
        "dm_family_sizes": {
            f"{target}/{loss}": int(len(group))
            for (target, loss), group in dm_pairs.groupby(["target", "loss"])
        },
        "publication_outputs": dict(publication_info),
        "environment": _environment(),
        "methodological_notes": [
            "Forecasts are evaluated as supplied; models are not retrained here.",
            "The primary loss series contains one value per complete daily 24-hour trajectory.",
            "Horizon-specific tests are secondary and use days as repeated observations.",
            "Exact duplicates are detected per target before formal pair construction.",
            "Holm adjustment is applied to target-unique model pairs within each target and loss.",
            "Descriptive metrics and trajectories retain every complete P-Q system.",
            "The configured regime split is descriptive sensitivity analysis, not a replacement test period.",
            "Future exogenous availability cannot be verified from prediction workbooks alone.",
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def run(config: RunConfig) -> Path:
    models, models_table, audit, skipped, duplicates = load_and_audit_models(config)
    unique_models, target_duplicates = target_unique_models(
        models,
        deduplicate=config.deduplicate,
    )
    enriched, overall, daily, regimes, sources = build_loss_tables(
        models,
        regime_split=config.regime_split,
    )
    formal_daily = pd.concat(
        [
            daily[
                (daily["target"] == target)
                & (daily["model_id"].isin(unique_models[target]))
            ]
            for target in TARGETS
        ],
        ignore_index=True,
    )
    formal_enriched = {
        target: {
            model_id: enriched[target][model_id]
            for model_id in unique_models[target]
        }
        for target in TARGETS
    }
    dm_pairs = build_daily_dm(
        formal_daily,
        hac_lag=config.hac_lag,
        min_days=config.min_days,
        alpha=config.alpha,
    )
    horizon_dm = (
        build_horizon_dm(
            formal_enriched,
            hac_lag=config.hac_lag,
            min_days=config.min_days,
            alpha=config.alpha,
            expected_hours=config.expected_hours,
        )
        if config.horizon_dm
        else pd.DataFrame()
    )
    league = build_league(dm_pairs, alpha=config.alpha)
    matrices = build_dm_matrices(dm_pairs)
    selection, retained_system = build_complete_system_selection(
        overall,
        selected_system=config.selected_system,
    )
    run_dir = _new_run_directory(Path(config.output_dir))
    if config.plots:
        create_plots(
            run_dir,
            enriched,
            overall,
            daily,
            dm_pairs,
            league,
            alpha=config.alpha,
            per_model=config.per_model_plots,
        )
    publication_info: Dict[str, Any] = {}
    if config.publication_figure:
        publication_info = create_publication_outputs(
            run_dir,
            dm_pairs,
            overall,
            target_duplicates,
            selection,
            retained_system=retained_system,
            dpi=config.publication_dpi,
        )
    write_outputs(
        run_dir,
        config,
        models_table,
        audit,
        skipped,
        duplicates,
        target_duplicates,
        sources,
        overall,
        daily,
        regimes,
        dm_pairs,
        horizon_dm,
        league,
        matrices,
        selection,
        retained_system,
        publication_info,
    )
    return run_dir


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=str(PROJECT_ROOT / "Proper Model Comparison" / "LGBM Evaluated"),
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--alignment", choices=("strict", "intersection"), default="strict")
    parser.add_argument("--expected-hours", type=int, default=24)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--hac-lag", default="auto", help="non-negative integer or 'auto'")
    parser.add_argument("--min-days", type=int, default=30)
    parser.add_argument("--regime-split", default="2022-02-24")
    parser.add_argument("--no-regime-split", action="store_true")
    parser.add_argument("--exclude-baselines", action="store_true")
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="retain exact paired and target-specific duplicate forecast vectors",
    )
    parser.add_argument("--no-horizon-dm", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-per-model-plots", action="store_true")
    parser.add_argument(
        "--no-publication-figure",
        action="store_true",
        help="skip the two-panel target-unique manuscript matrix",
    )
    parser.add_argument("--publication-dpi", type=int, default=600)
    parser.add_argument(
        "--selected-system",
        help=(
            "complete P-Q model identifier to highlight; by default the system "
            "with the lowest mean target-normalized RMSE is selected"
        ),
    )
    parser.add_argument("--max-models", type=int)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> RunConfig:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "DM Improved"
    if args.expected_hours < 1:
        raise ValueError("expected-hours must be positive")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if args.min_days < 5:
        raise ValueError("min-days must be at least 5")
    if args.hac_lag.lower() != "auto" and int(args.hac_lag) < 0:
        raise ValueError("hac-lag must be non-negative or 'auto'")
    if not 72 <= args.publication_dpi <= 1200:
        raise ValueError("publication-dpi must be between 72 and 1200")
    return RunConfig(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        start=args.start,
        end=args.end,
        alignment=args.alignment,
        expected_hours=args.expected_hours,
        alpha=args.alpha,
        hac_lag=args.hac_lag,
        min_days=args.min_days,
        regime_split=None if args.no_regime_split else args.regime_split,
        include_baselines=not args.exclude_baselines,
        deduplicate=not args.keep_duplicates,
        horizon_dm=not args.no_horizon_dm,
        plots=not args.no_plots,
        per_model_plots=not args.no_per_model_plots,
        max_models=args.max_models,
        publication_figure=not args.no_publication_figure,
        publication_dpi=args.publication_dpi,
        selected_system=args.selected_system,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    if not arguments:
        _parse_args(["--help"])
        return 0
    args = _parse_args(arguments)
    config = _config_from_args(args)
    if args.audit_only:
        models, model_table, audit, skipped, duplicates = load_and_audit_models(config)
        unique_models, target_duplicates = target_unique_models(
            models,
            deduplicate=config.deduplicate,
        )
        print(f"Valid distinct systems: {len(model_table)}")
        for target in TARGETS:
            print(
                f"{target}: {len(models[target])} complete systems, "
                f"{len(unique_models[target])} target-unique forecast vectors"
            )
        print(audit.to_string(index=False))
        if skipped:
            print("Skipped files:")
            for row in skipped:
                print(f"  {row['file']}: {row['reason']}")
        if duplicates:
            print("Exact paired duplicates removed:")
            for row in duplicates:
                print(f"  {row['removed_model']} -> {row['kept_model']}")
        if not target_duplicates.empty:
            print("Exact target-specific duplicates removed:")
            for row in target_duplicates.itertuples(index=False):
                print(
                    f"  {row.target}: {row.duplicate_model_removed} "
                    f"-> {row.representative_model}"
                )
        return 0

    run_dir = run(config)
    print(f"DM comparison completed: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

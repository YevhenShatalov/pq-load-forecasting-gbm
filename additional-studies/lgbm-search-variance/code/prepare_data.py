#!/usr/bin/env python3
"""Prepare hourly features, build nested split files, and audit all inputs.

Edit the MANUAL SETTINGS block and run this file from an IDE, or override the
same values from the command line.  Nothing is written unless EXECUTE is true
or ``--execute`` is supplied.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

_INTERNAL_DIR = Path(__file__).resolve().parent / "_internal"
if str(_INTERNAL_DIR) not in sys.path:
    sys.path.insert(0, str(_INTERNAL_DIR))

import pandas as pd

import audit_search_inputs
import build_nested_search_splits
import data_prep_and_features as features


# ---------------------------------------------------------------------------
# MANUAL SETTINGS: used when the script is launched without CLI overrides.
# ---------------------------------------------------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ACTION = "audit"  # data, splits, audit, or all
EXECUTE = False
OVERWRITE = False
SPLIT_BACKEND = "openpyxl"  # portable default; artifact-tool preserves legacy styling

# Required only for ACTION=data or ACTION=all.
P_2021: Path | None = None
P_2022: Path | None = None
Q_2021: Path | None = None
Q_2022: Path | None = None
WEATHER: Path | None = None
DAYLIGHT: Path | None = None
RAW_SHEET = "Sheet1"

TARGETS = ("P_Power", "Q_Power")
LAG_SHEETS = (1, 24, 168)


def build_prepared_workbooks(
    package_root: Path,
    raw_files: dict[str, Path | None],
    *,
    raw_sheet: str,
    overwrite: bool,
) -> tuple[Path, Path]:
    """Reproduce the prepared PQ and historical split workbooks."""
    missing = [name for name, path in raw_files.items() if path is None]
    if missing:
        raise ValueError(
            "Raw paths are required for data preparation: " + ", ".join(missing)
        )
    resolved = {name: Path(path).expanduser().resolve() for name, path in raw_files.items()}
    absent = [str(path) for path in resolved.values() if not path.exists()]
    if absent:
        raise FileNotFoundError("Raw input files not found:\n" + "\n".join(absent))

    input_dir = package_root / "Input"
    input_dir.mkdir(parents=True, exist_ok=True)
    pq_path = input_dir / "PQ.xlsx"
    splits_path = input_dir / "splits_historical.xlsx"
    existing = [path for path in (pq_path, splits_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to replace existing output. Use --overwrite after checking: "
            + ", ".join(str(path) for path in existing)
        )

    p21 = pd.read_excel(resolved["p_2021"], sheet_name=raw_sheet, usecols=[0, 1])
    p22 = pd.read_excel(resolved["p_2022"], sheet_name=raw_sheet, usecols=[0, 1])
    q21 = pd.read_excel(resolved["q_2021"], sheet_name=raw_sheet, usecols=[0, 1])
    q22 = pd.read_excel(resolved["q_2022"], sheet_name=raw_sheet, usecols=[0, 1])
    weather = pd.read_excel(resolved["weather"], sheet_name=raw_sheet)
    daylight = pd.read_excel(resolved["daylight"], sheet_name=raw_sheet)

    pq21 = pd.concat([p21, q21], axis=1, join="outer")
    pq22 = pd.concat([p22, q22], axis=1, join="outer")
    merged = pd.concat([pq21, pq22])
    merged = merged.loc[:, ~merged.columns.duplicated()].copy()
    merged = pd.merge(merged, weather, on="Datetime", how="inner")
    merged = pd.merge(
        merged, features.solar_features(daylight), on="Datetime", how="inner"
    )
    merged = features.add_season_features(
        merged, datetime_col="Datetime", hemisphere="north"
    )

    prepared = features.localize_fixed_utc2(
        merged, explicit_format="%Y-%m-%d %H:%M:%S"
    )
    prepared = pd.concat(
        [
            prepared.set_index("Datetime"),
            features.build_holiday_features(prepared.set_index("Datetime").index),
        ],
        axis=1,
        join="outer",
    )
    for target in TARGETS:
        prepared[target] = features.nan_outliers(prepared[target])

    p_filled = features.fill_nan_with_sarima(
        prepared.reset_index(names="Datetime"), "Datetime", "P_Power"
    )
    q_filled = features.fill_nan_with_sarima(
        prepared.reset_index(names="Datetime"), "Datetime", "Q_Power"
    )
    prepared["P_Power"] = pd.Series(
        p_filled["P_Power_filled"].to_numpy(), index=prepared.index
    )
    prepared["Q_Power"] = pd.Series(
        q_filled["Q_Power_filled"].to_numpy(), index=prepared.index
    )
    prepared.index = prepared.index.tz_localize(None)

    last_year_start = pd.Period(
        prepared.index[-1].normalize(), freq="Y"
    ).start_time.normalize()
    development_rows = prepared.index < last_year_start
    pq_sheets: dict[str, pd.DataFrame] = {}
    split_sheets: dict[str, pd.DataFrame] = {}
    active_lags: list[int] = []
    for lag in LAG_SHEETS:
        active_lags.append(lag)
        matrix = features.make_lag_features(
            prepared.copy(),
            targets=list(TARGETS),
            lags=active_lags,
            drop_original=False,
        )
        pq_sheets[str(lag)] = matrix.dropna()
        splits = features.rolling_dates(matrix.loc[development_rows].dropna())
        split_sheets[str(lag)] = pd.DataFrame(splits).drop(
            columns="m", errors="ignore"
        )

    with pd.ExcelWriter(pq_path, engine="openpyxl", mode="w") as writer:
        for sheet_name, frame in pq_sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name)
    with pd.ExcelWriter(splits_path, engine="openpyxl", mode="w") as writer:
        for sheet_name, frame in split_sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"[prepared] {pq_path}")
    print(f"[prepared] {splits_path}")
    return pq_path, splits_path


def audit_inputs(package_root: Path) -> None:
    input_dir = package_root / "Input"
    audit_search_inputs.audit_inputs(
        package_root=package_root,
        pq_path=input_dir / "PQ.xlsx",
        search_path=input_dir / "splits_search24.xlsx",
        search36_path=input_dir / "splits_search36.xlsx",
        search48_path=input_dir / "splits_search48.xlsx",
        gap_path=input_dir / "splits_gap48.xlsx",
        event_path=input_dir / "splits_event16.xlsx",
        historical_p_meta=input_dir / "historical_source_mut_l24_P.meta.json",
        historical_q_meta=input_dir / "historical_source_mut_l24_Q.meta.json",
        original_splits_path=input_dir / "splits_historical.xlsx",
    )
    print(
        f"[audit passed] "
        f"{package_root / 'Results' / 'reproducibility' / 'input_audit.json'}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=PACKAGE_ROOT)
    parser.add_argument("--action", choices=("data", "splits", "audit", "all"), default=ACTION)
    parser.add_argument("--execute", action=argparse.BooleanOptionalAction, default=EXECUTE)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=OVERWRITE)
    parser.add_argument(
        "--split-backend",
        choices=("openpyxl", "artifact-tool"),
        default=SPLIT_BACKEND,
    )
    parser.add_argument("--p-2021", type=Path, default=P_2021)
    parser.add_argument("--p-2022", type=Path, default=P_2022)
    parser.add_argument("--q-2021", type=Path, default=Q_2021)
    parser.add_argument("--q-2022", type=Path, default=Q_2022)
    parser.add_argument("--weather", type=Path, default=WEATHER)
    parser.add_argument("--daylight", type=Path, default=DAYLIGHT)
    parser.add_argument("--raw-sheet", default=RAW_SHEET)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = _parser().parse_args(argv)
    package_root = args.package_root.expanduser().resolve()
    print(f"Preparation action: {args.action}")
    print(f"Package root:       {package_root}")
    if not args.execute:
        print("Preview only. Set EXECUTE=True or add --execute to write files.")
        return 0

    raw_files = {
        "p_2021": args.p_2021,
        "p_2022": args.p_2022,
        "q_2021": args.q_2021,
        "q_2022": args.q_2022,
        "weather": args.weather,
        "daylight": args.daylight,
    }
    if args.action in {"data", "all"}:
        build_prepared_workbooks(
            package_root,
            raw_files,
            raw_sheet=args.raw_sheet,
            overwrite=bool(args.overwrite),
        )
    if args.action in {"splits", "all"}:
        build_nested_search_splits.build_nested_splits(
            package_root, backend=args.split_backend
        )
    if args.action in {"audit", "all"}:
        audit_inputs(package_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run leakage checks, statistical comparisons, and publication outputs.

``final`` is the article workflow.  The other plans retain the broader search-
mechanism analysis and the legacy workbook-based all-model DM comparison.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

_INTERNAL_DIR = Path(__file__).resolve().parent / "_internal"
if str(_INTERNAL_DIR) not in sys.path:
    sys.path.insert(0, str(_INTERNAL_DIR))

import analyze_search
import analyze_t60_breadth
import build_conference_evidence
import dm_across_models


# ---------------------------------------------------------------------------
# MANUAL SETTINGS
# ---------------------------------------------------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PLAN = "final"  # final, breadth, full-search, conference, workbook-dm, or all
EXECUTE = False
THREADS = 8
STRUCTURAL_ONLY_AUDIT = False
BUILD_RESULTS_WORKBOOK = False

HAC_LAG = "3"
MIN_DAYS = 30
REQUIRE_ALL_SEARCH_SYSTEMS = True

DM_INPUT_DIR: Path | None = None
DM_OUTPUT_DIR: Path | None = None
DM_START = "2022-01-01 00:00:00"
DM_END = "2022-02-23 23:00:00"
DM_SELECTED_SYSTEM: str | None = None


def _run_leakage_audit(
    package_root: Path, *, threads: int, structural_only: bool
) -> None:
    command = [
        sys.executable,
        str(_INTERNAL_DIR / "audit_59day_forecasts.py"),
        "--package-root",
        str(package_root),
        "--threads",
        str(threads),
    ]
    if structural_only:
        command.append("--structural-only")
    subprocess.run(command, check=True, cwd=package_root)


def _build_workbook(package_root: Path) -> None:
    node = os.environ.get("NODE_EXECUTABLE", "node")
    subprocess.run(
        [
            node,
            str(_INTERNAL_DIR / "build_full_results.mjs"),
            str(package_root / "Results"),
        ],
        check=True,
        cwd=package_root,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=PACKAGE_ROOT)
    parser.add_argument(
        "--plan",
        choices=("final", "breadth", "full-search", "conference", "workbook-dm", "all"),
        default=PLAN,
    )
    parser.add_argument("--execute", action=argparse.BooleanOptionalAction, default=EXECUTE)
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--structural-only-audit", action=argparse.BooleanOptionalAction, default=STRUCTURAL_ONLY_AUDIT)
    parser.add_argument("--build-results-workbook", action=argparse.BooleanOptionalAction, default=BUILD_RESULTS_WORKBOOK)
    parser.add_argument("--hac-lag", default=HAC_LAG)
    parser.add_argument("--min-days", type=int, default=MIN_DAYS)
    parser.add_argument("--require-all-search-systems", action=argparse.BooleanOptionalAction, default=REQUIRE_ALL_SEARCH_SYSTEMS)
    parser.add_argument("--dm-input-dir", type=Path, default=DM_INPUT_DIR)
    parser.add_argument("--dm-output-dir", type=Path, default=DM_OUTPUT_DIR)
    parser.add_argument("--dm-start", default=DM_START)
    parser.add_argument("--dm-end", default=DM_END)
    parser.add_argument("--dm-selected-system", default=DM_SELECTED_SYSTEM)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = _parser().parse_args(argv)
    package_root = args.package_root.expanduser().resolve()
    steps: list[str] = []
    if args.plan in {"final", "breadth", "all"}:
        steps.append("T60 breadth metrics, paired tests, tables, and figures")
    if args.plan == "final":
        steps.append("supporting search summaries for the compact result package")
    if args.plan in {"full-search", "all"}:
        steps.append("full search-mechanism statistical analysis")
    if args.plan in {"final", "conference", "all"}:
        steps += ["59-origin structural/behavioral leakage audit", "publication evidence package"]
    if args.plan in {"workbook-dm", "all"}:
        steps.append("legacy workbook-based all-model DM comparison")
    if args.build_results_workbook:
        steps.append("formatted full-results workbook")

    print(f"Analysis plan: {args.plan}")
    for number, step in enumerate(steps, start=1):
        print(f"  {number}. {step}")
    if not args.execute:
        print("Preview only. Set EXECUTE=True or add --execute to generate outputs.")
        return 0

    if args.plan in {"final", "breadth", "all"}:
        analyze_t60_breadth.run(package_root)
    if args.plan in {"final", "full-search", "all"}:
        analyze_search.run_analysis(
            package_root=package_root,
            require_all=(
                False
                if args.plan == "final"
                else bool(args.require_all_search_systems)
            ),
            hac_lag=str(args.hac_lag),
            min_days=int(args.min_days),
        )
    if args.plan in {"final", "conference", "all"}:
        _run_leakage_audit(
            package_root,
            threads=max(1, int(args.threads)),
            structural_only=bool(args.structural_only_audit),
        )
        if args.structural_only_audit:
            print("[skipped] Conference evidence requires the full behavioral audit.")
        else:
            build_conference_evidence.main(package_root)
    if args.plan in {"workbook-dm", "all"}:
        input_dir = args.dm_input_dir or package_root / "_work" / "probabilistic"
        output_dir = args.dm_output_dir or package_root / "_work" / "probabilistic_dm_primary"
        dm_args = [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--start",
            args.dm_start,
            "--end",
            args.dm_end,
            "--hac-lag",
            str(args.hac_lag),
        ]
        if args.dm_selected_system:
            dm_args += ["--selected-system", args.dm_selected_system]
        status = dm_across_models.main(dm_args)
        if status:
            return int(status)
    if args.build_results_workbook:
        _build_workbook(package_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

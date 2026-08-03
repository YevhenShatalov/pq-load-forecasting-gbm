#!/usr/bin/env python3
"""Refit retained specifications and evaluate complete 24-hour forecasts.

Technical forecast files are written below ``_work``. The reader-facing
package in ``Results`` is assembled by ``compare_models.py``. Point forecasts
and fixed Q4 residual-trajectory probabilistic scores are selected
independently.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

_INTERNAL_DIR = Path(__file__).resolve().parent / "_internal"
if str(_INTERNAL_DIR) not in sys.path:
    sys.path.insert(0, str(_INTERNAL_DIR))

import Forecasting_search as training
import evaluate_search as point
import probabilistic_analysis as probabilistic


# ---------------------------------------------------------------------------
# MANUAL SETTINGS
# ---------------------------------------------------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PLAN = "final"  # final, depth, breadth, replication, or custom
EXECUTE = False
PERIODS = ("q4_2021", "prewar_2022", "late_february_2022")
RUN_PROBABILISTIC = True
INCLUDE_BASELINES = True
DEVICE = "cpu"
THREADS = 4
MAX_ORIGINS: int | None = None
ENGINE_DRY_RUN = False

# Used by PLAN=custom.
MODELS: tuple[str, ...] = ()
OUTPUT_NAMESPACE: Path | None = None
CHECKPOINT_NAMESPACE: str | None = None

# Used by PLAN=replication.
SAMPLER_SEED = 17
MILESTONE = 60


def _stages(args: argparse.Namespace) -> list[dict[str, Any]]:
    depth = {
        "name": "search depth",
        "models": ["sx_hist", "sx_norm_t10", "sx_norm_t30", "sx_norm_t60"],
        "output_namespace": None,
        "checkpoint_namespace": None,
        "probabilistic_output": Path("_work/probabilistic"),
    }
    breadth = {
        "name": "T60 validation breadth",
        "models": [
            "sx_hist",
            "sx_norm_t60",
            "sx_norm_s36_t60",
            "sx_norm_s48_t60",
        ],
        "output_namespace": Path("breadth_t60/point"),
        "checkpoint_namespace": "breadth_t60",
        "probabilistic_output": Path("_work/breadth_t60/probabilistic"),
    }
    if args.plan == "final":
        return [depth, breadth]
    if args.plan == "depth":
        return [depth]
    if args.plan == "breadth":
        return [breadth]
    if args.plan == "replication":
        snapshot = (
            f"sx_norm_t{args.milestone}"
            if args.sampler_seed == 42
            else f"sx_norm_s{args.sampler_seed}_t{args.milestone}"
        )
        base = Path("replications") / f"seed_{args.sampler_seed}" / f"t{args.milestone}"
        return [
            {
                "name": f"seed {args.sampler_seed}, T{args.milestone}",
                "models": [snapshot],
                "output_namespace": base / "forecasting",
                "checkpoint_namespace": f"seed_{args.sampler_seed}_t{args.milestone}",
                "probabilistic_output": Path("_work") / base / "probabilistic",
            }
        ]
    if not args.models:
        raise ValueError("PLAN=custom requires MODELS or --models")
    return [
        {
            "name": "custom",
            "models": list(args.models),
            "output_namespace": args.output_namespace,
            "checkpoint_namespace": args.checkpoint_namespace,
            "probabilistic_output": (
                Path("_work") / args.output_namespace.parent / "probabilistic"
                if args.output_namespace is not None
                else Path("_work/probabilistic")
            ),
        }
    ]


def _parser() -> argparse.ArgumentParser:
    period_choices = (*point.PERIODS, "event16", "event16_gaps")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=PACKAGE_ROOT)
    parser.add_argument("--plan", choices=("final", "depth", "breadth", "replication", "custom"), default=PLAN)
    parser.add_argument("--execute", action=argparse.BooleanOptionalAction, default=EXECUTE)
    parser.add_argument("--periods", nargs="+", choices=period_choices, default=list(PERIODS))
    parser.add_argument("--probabilistic", action=argparse.BooleanOptionalAction, default=RUN_PROBABILISTIC)
    parser.add_argument("--include-baselines", action=argparse.BooleanOptionalAction, default=INCLUDE_BASELINES)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default=DEVICE)
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--max-origins", type=int, default=MAX_ORIGINS)
    parser.add_argument("--engine-dry-run", action=argparse.BooleanOptionalAction, default=ENGINE_DRY_RUN)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--output-namespace", type=Path, default=OUTPUT_NAMESPACE)
    parser.add_argument("--checkpoint-namespace", default=CHECKPOINT_NAMESPACE)
    parser.add_argument("--sampler-seed", type=int, default=SAMPLER_SEED)
    parser.add_argument("--milestone", type=int, choices=(10, 30, 60), default=MILESTONE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = _parser().parse_args(argv)
    package_root = args.package_root.expanduser().resolve()
    stages = _stages(args)
    print(f"Evaluation plan: {args.plan}")
    print(f"Periods:         {', '.join(args.periods)}")
    for number, stage in enumerate(stages, start=1):
        point_output = (
            Path("_work") / stage["output_namespace"]
            if stage["output_namespace"] is not None
            else Path("_work/forecasting")
        )
        print(f"  {number}. {stage['name']}: {', '.join(stage['models'])}")
        print(f"     point output: {point_output.as_posix()}")
        if args.probabilistic:
            print(
                "     probabilistic output: "
                f"{stage['probabilistic_output'].as_posix()}"
            )
    if not args.execute:
        print("Preview only. Set EXECUTE=True or add --execute to refit/evaluate models.")
        return 0
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if args.probabilistic and "q4_2021" not in args.periods:
        raise ValueError("Probabilistic calibration requires q4_2021 in --periods")
    if args.probabilistic and (args.max_origins is not None or args.engine_dry_run):
        raise ValueError("Disable probabilistic outputs for partial or dry-run evaluation")

    use_gpu = training._resolve_experiment_device(
        package_root=package_root,
        requested=args.device,
        threads=args.threads,
    )
    for stage in stages:
        point.run_evaluation(
            package_root=package_root,
            periods=list(args.periods),
            requested_models=stage["models"],
            include_baselines=bool(args.include_baselines),
            use_gpu=use_gpu,
            threads=args.threads,
            max_origins=args.max_origins,
            dry_run=bool(args.engine_dry_run),
            output_namespace=stage["output_namespace"],
            checkpoint_namespace=stage["checkpoint_namespace"],
        )
        if args.probabilistic:
            point_dir = (
                package_root / "_work" / stage["output_namespace"]
                if stage["output_namespace"] is not None
                else package_root / "_work" / "forecasting"
            )
            snapshots = list(stage["models"])
            if args.include_baselines:
                snapshots += ["SNaive-24", "SNaive-168"]
            probabilistic.run(
                package_root,
                predictions_path=point_dir / "predictions.csv",
                output_dir=package_root / stage["probabilistic_output"],
                snapshots=snapshots,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

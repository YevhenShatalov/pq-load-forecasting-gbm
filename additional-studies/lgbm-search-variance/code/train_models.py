#!/usr/bin/env python3
"""Run or resume the LightGBM/Optuna training experiment.

The MANUAL SETTINGS block is intended for IDE use.  Every setting also has a
command-line equivalent.  Preview is the default so an accidental launch does
not restart a long search; use ``--execute`` only after reviewing the plan.
"""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Sequence

_INTERNAL_DIR = Path(__file__).resolve().parent / "_internal"
if str(_INTERNAL_DIR) not in sys.path:
    sys.path.insert(0, str(_INTERNAL_DIR))

import Forecasting_search as engine


# ---------------------------------------------------------------------------
# MANUAL SETTINGS
# ---------------------------------------------------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PLAN = "final"  # final, single, all-methods, inspect, protocol-smoke, benchmark, replay-pruned
EXECUTE = False

JOB = "norm"  # raw, norm, norm36, norm48, random, gap, alt, or all
SAMPLER_SEEDS = (42, 17, 73)
TERMINAL_BUDGET = 60
DEVICE = "cpu"  # auto, cpu, or gpu
THREADS = 4
FOLD_LIMIT: int | None = None
ENGINE_DRY_RUN = False
ISOLATED_SMOKE_RUN = False

BENCHMARK_FOLDS = 6
PROTOCOL_SMOKE_FOLDS = 3
REPLAY_JOB = "norm"
TRIALS_PER_TARGET_CYCLE: int | None = None
OUTER_CYCLES: int | None = None


def _run_arguments(args: argparse.Namespace) -> list[list[str]]:
    root = str(args.package_root.expanduser().resolve())
    common = ["--package-root", root]
    commands: list[list[str]] = []

    if args.plan == "protocol-smoke":
        return [common + ["protocol-smoke", "--threads", str(args.threads), "--folds", str(args.protocol_smoke_folds)]]
    if args.plan == "benchmark":
        return [common + ["benchmark", "--threads", str(args.threads), "--folds", str(args.benchmark_folds)]]
    if args.plan == "replay-pruned":
        command = common + [
            "replay-pruned",
            "--job",
            args.replay_job,
            "--device",
            args.device,
            "--threads",
            str(args.threads),
        ]
        seed = args.sampler_seeds[0] if args.sampler_seeds else 42
        if seed != 42:
            command += ["--sampler-seed", str(seed)]
        return [command]

    if args.plan == "final":
        if args.isolated_smoke_run:
            raise ValueError("Use plan=single for an isolated smoke run.")
        jobs = ["norm", "norm36", "norm48"]
        runs = [(job, seed) for seed in args.sampler_seeds for job in jobs]
    elif args.plan == "all-methods":
        runs = [("all", 42), ("norm36", 42), ("norm48", 42)]
    elif args.plan == "inspect":
        runs = [("all", 42), ("norm36", 42), ("norm48", 42)]
    else:
        runs = (
            [(args.job, 42)]
            if args.job in {"alt", "all"}
            else [(args.job, seed) for seed in args.sampler_seeds]
        )

    for job, seed in runs:
        command = common + [
            "run",
            "--job",
            job,
            "--device",
            args.device,
            "--threads",
            str(args.threads),
        ]
        if job != "alt" and job != "all" and seed != 42:
            command += ["--sampler-seed", str(seed)]
        if args.terminal_budget is not None and job not in {"alt", "all"}:
            command += ["--terminal-budget", str(args.terminal_budget)]
        if args.trials_per_target_cycle is not None and job in {"alt", "all"}:
            command += ["--trials-per-target-cycle", str(args.trials_per_target_cycle)]
        if args.outer_cycles is not None and job in {"alt", "all"}:
            command += ["--outer-cycles", str(args.outer_cycles)]
        if args.fold_limit is not None:
            command += ["--fold-limit", str(args.fold_limit)]
        if args.isolated_smoke_run:
            command.append("--smoke")
        if args.engine_dry_run or args.plan == "inspect":
            command.append("--dry-run")
        commands.append(command)
    return commands


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=PACKAGE_ROOT)
    parser.add_argument(
        "--plan",
        choices=("final", "single", "all-methods", "inspect", "protocol-smoke", "benchmark", "replay-pruned"),
        default=PLAN,
    )
    parser.add_argument("--execute", action=argparse.BooleanOptionalAction, default=EXECUTE)
    parser.add_argument(
        "--job",
        choices=("raw", "norm", "norm36", "norm48", "random", "gap", "alt", "all"),
        default=JOB,
    )
    parser.add_argument("--sampler-seeds", nargs="+", type=int, default=list(SAMPLER_SEEDS))
    parser.add_argument("--terminal-budget", type=int, default=TERMINAL_BUDGET)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default=DEVICE)
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--fold-limit", type=int, default=FOLD_LIMIT)
    parser.add_argument("--engine-dry-run", action=argparse.BooleanOptionalAction, default=ENGINE_DRY_RUN)
    parser.add_argument("--isolated-smoke-run", action=argparse.BooleanOptionalAction, default=ISOLATED_SMOKE_RUN)
    parser.add_argument("--benchmark-folds", type=int, default=BENCHMARK_FOLDS)
    parser.add_argument("--protocol-smoke-folds", type=int, default=PROTOCOL_SMOKE_FOLDS)
    parser.add_argument("--replay-job", choices=("norm", "random"), default=REPLAY_JOB)
    parser.add_argument("--trials-per-target-cycle", type=int, default=TRIALS_PER_TARGET_CYCLE)
    parser.add_argument("--outer-cycles", type=int, default=OUTER_CYCLES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = _parser().parse_args(argv)
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if not args.sampler_seeds:
        raise ValueError("At least one sampler seed is required")
    commands = _run_arguments(args)
    print(f"Training plan: {args.plan}")
    for number, command in enumerate(commands, start=1):
        operation = next(
            index
            for index, token in enumerate(command)
            if token in {"run", "protocol-smoke", "benchmark", "replay-pruned"}
        )
        print(f"  {number:02d}. {shlex.join(command[operation:])}")
    if not args.execute:
        print("Preview only. Set EXECUTE=True or add --execute to run/resume training.")
        return 0

    for command in commands:
        status = engine.main(command)
        if status:
            return int(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

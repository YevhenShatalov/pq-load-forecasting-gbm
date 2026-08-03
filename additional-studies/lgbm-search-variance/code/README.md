# Code

[Back to the package README](../README.md)

The complete workflow has four public programs. These are the only Python
files a reader normally needs to operate.

| Program | Purpose | Main output |
|---|---|---|
| `prepare_data.py` | Prepare or audit the hourly data and chronological splits. | `../Input` and input-audit records. |
| `train_models.py` | Run or resume the registered LightGBM/Optuna studies. | `../_work` checkpoints and retained specifications. |
| `evaluate_models.py` | Refit retained specifications at rolling origins and generate 24-hour point and probabilistic forecasts. | Technical evaluation files below `../_work`. |
| `compare_models.py` | Run leakage checks, calculate metrics and paired tests, and assemble publication outputs. | The compact `../Results` package. |

The `_internal` directory contains the validated implementation called by the
four public programs. It is included for transparency and reproduction, but
its modules are not separate user-facing workflow steps.

## Installation

From the package root:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r code/requirements.txt
```

The final experiment requires LightGBM. The shared internal engine retains
optional XGBoost and CatBoost support from the broader project, but those
learners are not required for the reported LightGBM experiment. Node.js is
needed only when `--build-results-workbook` is requested.

## Two equivalent interfaces

Each public program exposes the same choices in two ways:

1. Edit the constants under `MANUAL SETTINGS` and run the file from an IDE or
   Colab.
2. Leave the file unchanged and pass command-line arguments.

Command-line arguments override the manual defaults. Use
`python code/PROGRAM.py --help` to list every option.

All four programs are preview-only by default. A preview prints the selected
work but does not start training or write output. Set `EXECUTE = True` in the
file or add `--execute` only after checking the plan.

## 1. Prepare or audit data

The publication package already includes the prepared workbook and all split
files. The normal reproducibility action is therefore the audit:

```bash
python code/prepare_data.py --action audit --execute
```

Available actions:

| Action | Behavior |
|---|---|
| `audit` | Validate `PQ.xlsx`, split nesting, date boundaries, scales, and historical-anchor metadata. |
| `splits` | Rebuild the 36- and 48-origin nested split workbooks from the 24-origin design. |
| `data` | Recreate `PQ.xlsx` and `splits_historical.xlsx` from six user-supplied raw workbooks. |
| `all` | Run data preparation, split creation, and the audit in sequence. |

Raw-data preparation requires explicit paths:

```bash
python code/prepare_data.py --action data \
  --p-2021 PATH --p-2022 PATH \
  --q-2021 PATH --q-2022 PATH \
  --weather PATH --daylight PATH \
  --execute
```

Existing prepared workbooks are protected by default. Add `--overwrite` only
when replacement is intentional. The portable split writer is `openpyxl`, the
default backend.

## 2. Train or resume models

Preview the publication plan:

```bash
python code/train_models.py --plan final
```

Run it after checking the nine printed jobs:

```bash
python code/train_models.py --plan final --device cpu --threads 4 --execute
```

The `final` plan runs the normalized 24-, 36-, and 48-origin studies for
sampler seeds 42, 17, and 73. The 24-origin study exports nested 10-, 30-, and
60-trial milestones; the wider studies export the 60-trial milestone.

To run one study only:

```bash
python code/train_models.py --plan single --job norm \
  --sampler-seeds 42 --terminal-budget 60 \
  --device cpu --threads 4 --execute
```

Publication jobs:

| Job | Split design | Objective | Milestones |
|---|---|---|---|
| `norm` | 24 origins | Fixed-scale normalized paired RMSE | 10, 30, 60 |
| `norm36` | 36 nested origins | Same objective | 60 |
| `norm48` | 48 nested origins | Same objective | 60 |

The additional `raw`, `random`, `gap`, and `alt` jobs preserve exploratory
experiments from the broader search-mechanism study. They are not needed to
rebuild the compact publication results.

Training is checkpointed. Rerun the identical command to continue an
interrupted study. A checkpoint is accepted only when its structural
fingerprint matches the requested job. Do not run two processes with the same
job and sampler seed concurrently.

## 3. Evaluate retained specifications

Run the complete publication evaluation:

```bash
python code/evaluate_models.py --plan final \
  --device cpu --threads 4 --execute
```

The default periods are:

- `q4_2021`, used to construct 92 complete calibration residual trajectories;
- `prewar_2022`, the 54-origin primary interval through 23 February; and
- `late_february_2022`, the five-origin descriptive interval.

Evaluation does not merely reuse predictions stored in a tuning model. At
every midnight origin, it reads the retained specification, refits separate P
and Q estimators using rows available through the preceding hour, and produces
one synchronized recursive 24-hour trajectory. Future targets are masked.

Useful plans:

| Plan | Systems evaluated |
|---|---|
| `depth` | Historical anchor plus the 24-origin T10, T30, and T60 milestones. |
| `breadth` | Historical anchor plus the 24-, 36-, and 48-origin T60 systems. |
| `final` | Both depth and breadth sets. |
| `replication` | One selected sampler-seed and milestone replication. |
| `custom` | Explicit snapshot identifiers supplied with `--models`. |

Probabilistic evaluation is enabled by default and requires `q4_2021` because
the empirical residual ensemble is calibrated there. Disable it for a partial
smoke run:

```bash
python code/evaluate_models.py --plan depth --max-origins 1 \
  --no-probabilistic --execute
```

## 4. Compare models and build outputs

After evaluation completes, generate the compact result package:

```bash
python code/compare_models.py --plan final --execute
```

The final plan:

- calculates depth and breadth summaries;
- generates primary and descriptive metrics;
- runs the planned daily paired tests and Holm adjustments;
- performs the 59-origin structural and behavioral leakage audit;
- creates the compact CSV tables and complete PNG/PDF figures.

The formatted Excel workbook is optional:

```bash
python code/compare_models.py --plan final \
  --build-results-workbook --execute
```

Other plans expose the broader search analysis, conference evidence builder,
and legacy workbook-based all-model Diebold-Mariano comparison. They are not
required for the main GitHub result package.

## Complete run order

```bash
python code/_internal/verify_setup.py
python code/prepare_data.py --action audit --execute
python code/train_models.py --plan final --execute
python code/evaluate_models.py --plan final --execute
python code/compare_models.py --plan final --execute
```

Training and evaluation write checkpoints, logs, and per-origin caches below
`../_work`. This directory is ignored by Git. Only `compare_models.py` writes
the compact reader-facing files under `../Results`.

## Source verification

Run this no-training check from the package root:

```bash
python code/_internal/verify_setup.py
```

It verifies `SOURCE_PROVENANCE.json`, compiles all public and internal Python
files, and rebuilds the nested 36- and 48-origin split designs in a temporary
directory. It does not fit or evaluate a model.


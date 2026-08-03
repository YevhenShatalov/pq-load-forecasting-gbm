# Additional Studies

This directory contains self-contained experiments that extend the main
P-Q forecasting study without changing the repository's existing `src`,
`data`, `models`, or `results` structure.

Each study has its own code, inputs, model artifacts, result package, and
reproducibility documentation. Results from one study should not be mixed with
the primary repository results unless the corresponding protocol explicitly
supports that comparison.

## Included study

### LightGBM search depth and validation breadth

[`lgbm-search-variance`](lgbm-search-variance/README.md) investigates how a
fixed paired active/reactive-power LightGBM forecasting architecture responds
to two allocations of hyperparameter-tuning effort:

- search depth at 10, 30, and 60 completed Optuna trials on 24 chronological
  validation origins; and
- validation breadth at 24, 36, and 48 nested chronological origins with the
  trial budget fixed at 60.

The study includes three Optuna sampler seeds, leakage-safe rolling-origin
evaluation, point and probabilistic accuracy, paired daily statistical tests,
publication figures, and hash-based manifests. The five systems used in the
2022 forecast comparison are frozen seed-42 specifications; the other seeds
describe development-search sensitivity.

Start with the study's [main README](lgbm-search-variance/README.md). Its
folder-specific guides document the public programs, authoritative inputs,
retained LightGBM artifacts, and compact result files.

## Organization rule

Future supplementary experiments should be added as sibling directories and
must provide their own top-level README. They should identify:

- the scientific question and fixed assumptions;
- which data and forecast intervals are used;
- which variables are changed experimentally;
- the exact reproduction commands;
- the main result and statistical interpretation; and
- any limitations that affect comparison with the primary study.

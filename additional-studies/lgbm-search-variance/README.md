# Optuna Search Depth and Validation Breadth for P-Q Forecasting

This is the reproducibility package for the study:

> **Effect of Optuna Search Depth and Temporal Validation Breadth on Paired
> Active and Reactive Power Forecasting**

The experiment uses one fixed LightGBM architecture for 24-hour substation
forecasting and asks how tuning effort should be divided between:

- **search depth**, represented by 10, 30, and 60 completed Optuna trials on
  the same 24 chronological validation origins; and
- **validation breadth**, represented by 24, 36, and 48 nested chronological
  origins at the same 60-trial budget.

The package contains the prepared data, split definitions, retained model
artifacts, complete public code, compact result data, tables, figures, and
hash-based reproducibility records. It can be inspected without retraining.

## Experiment at a glance

Every learned configuration keeps the forecasting architecture fixed:

- separate LightGBM estimators for active power (`P`, kW) and reactive power
  (`Q`, kVAr);
- synchronized recursive prediction for the next 24 hours;
- mutual P-Q history with explicit lags of 1 and 24 hours;
- trailing means and sample standard deviations over 24 and 168 hours;
- the same eligible weather and calendar predictors;
- one shared LightGBM hyperparameter vector for the P and Q specifications;
- no target observation after the forecast origin;
- future weather and calendar rows treated as available at the origin.

Configuration names use `Sx-Ty`:

| Label | Meaning |
|---|---|
| `S24-T10` | 24 tuning origins; best eligible vector after 10 trials. |
| `S24-T30` | 24 tuning origins; best eligible vector after 30 trials. |
| `S24-T60` | 24 tuning origins; best eligible vector after 60 trials. |
| `S36-T60` | 36 tuning origins; best eligible vector after 60 trials. |
| `S48-T60` | 48 tuning origins; best eligible vector after 60 trials. |

`S24-T10`, `S24-T30`, and `S24-T60` are nested checkpoints from one continuous
seed-42 Optuna study, not independent restarts. Seed 42 supplies all five
specifications used in the 2022 forecast comparison. Seeds 17 and 73 describe
development-search sensitivity only.

## Evaluation design

Each retained specification is refitted at every midnight origin using all
prepared rows available through 23:00 of the preceding day. P and Q are then
forecast recursively for 24 hours. Both feature rows are formed from the same
pre-update history, both forecasts are generated, and only then is the pair
appended to history.

The external comparison contains:

- **54 primary origins:** 1 January through 23 February 2022;
- **5 descriptive origins:** 24 through 28 February 2022;
- **92 calibration trajectories:** 1 October through 31 December 2021 for the
  fixed empirical residual ensemble used in probabilistic scoring.

Daily paired tests use normalized squared-error and CRPS losses, a
Bartlett-Newey-West HAC variance estimate with lag 3, and separate Holm
corrections for the three search-depth and three validation-breadth contrasts.

## Main findings

On the 54-day primary interval, `S36-T60` is the numerical leader under both
the fixed-scale paired RMSE score and paired normalized CRPS. Its active-power
RMSE is 118.04 kW and its reactive-power RMSE is 115.24 kVAr. `S48-T60` has a
slightly lower Q RMSE of 115.13 kVAr, but not the best paired score.

Within the nested 24-origin search trajectory, `S24-T60` has significantly
lower paired CRPS than `S24-T10` and `S24-T30` after Holm correction. The
corresponding point-loss differences do not remain significant. The three
60-trial validation-breadth systems are not statistically separated after
Holm correction. The result therefore supports a broad near-optimal region,
not a universal optimum at one trial or fold count.

The complete values are in
[`Results/tables/Table2_primary_performance.csv`](Results/tables/Table2_primary_performance.csv)
and the paired tests are in
[`Results/tables/Table3_paired_tests.csv`](Results/tables/Table3_paired_tests.csv).

## Package layout

```text
structured/
|-- code/       Four public workflow programs and preserved internal modules
|-- Input/      Prepared data, time splits, scales, and input manifests
|-- Models/     Retained LightGBM snapshots, metadata, and model manifests
|-- Results/    Compact data, tables, figures, and reproducibility records
|-- .gitattributes
|-- .gitignore
`-- README.md
```

Detailed guides:

- [Code and command-line workflow](code/README.md)
- [Input data and split definitions](Input/README.md)
- [Model artifact naming and provenance](Models/README.md)
- [Result files and interpretation](Results/README.md)

Long-running checkpoints, per-origin caches, and technical traces are written
to `_work/`. That directory is reproducible scratch space and is ignored by
Git. It is not required to inspect the archived results.

## Requirements

- Python 3.13 was used for the final smoke tests.
- Dependencies are pinned by compatible version ranges in
  [`code/requirements.txt`](code/requirements.txt).
- Git LFS is required for the retained model files and Excel workbooks.
- Node.js is needed only to rebuild the optional formatted Excel result
  workbook; the scientific CSV files, tests, and figures are generated in
  Python.

After cloning the repository, run from this directory:

```bash
git lfs install
git lfs pull
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r code/requirements.txt
```

Activate the environment using the command appropriate for your operating
system before running the workflow.

## Quick verification

Verify source hashes, compile the Python files, and smoke-test nested split
creation without fitting a model:

```bash
python code/_internal/verify_setup.py
```

Audit the prepared data and registered split files:

```bash
python code/prepare_data.py --action audit --execute
```

The public programs are preview-only by default. For example, this prints the
complete training plan without launching Optuna:

```bash
python code/train_models.py --plan final
```

## Full reproduction

The archived tables and figures can be read directly from `Results`. A full
rebuild is computationally expensive. To reproduce training, rolling
evaluation, statistical comparisons, and publication outputs, run:

```bash
python code/train_models.py --plan final --execute
python code/evaluate_models.py --plan final --execute
python code/compare_models.py --plan final --execute
```

Training resumes from compatible checkpoints below `_work`; rerunning the same
command does not intentionally restart a completed study. Do not run two
training processes with the same job and seed at the same time.

To rebuild the optional formatted workbook as well:

```bash
python code/compare_models.py --plan final --build-results-workbook --execute
```

## Reproducibility records

- `Input/INPUT_MANIFEST.json` records hashes and workbook structure for every
  authoritative input.
- `Models/MODEL_MANIFEST.json` maps publication names to original snapshots
  and records duplicate artifacts.
- `code/_internal/SOURCE_PROVENANCE.json` records the source of each preserved
  implementation module.
- `Results/reproducibility/final_results_manifest.json` records SHA-256 hashes
  for the compact result package.
- `Results/reproducibility/forecast_leakage_audit.json` records structural and
  behavioral checks of the 59-origin recursive forecasts.

## Scope

The results describe one substation and one prepared hourly data set. Target
repair was completed before this experiment, and future weather rows are
treated as known. These conditions apply equally to every compared
configuration, but they limit external interpretation. The final five
February origins are descriptive and are not included in ordinary inference.

## Authors

- Yevhen Shatalov, National Technical University of Ukraine "Igor Sikorsky
  Kyiv Polytechnic Institute" ([ORCID](https://orcid.org/0009-0003-5505-5674))
- Valerii Kyryk, National Technical University of Ukraine "Igor Sikorsky Kyiv
  Polytechnic Institute"


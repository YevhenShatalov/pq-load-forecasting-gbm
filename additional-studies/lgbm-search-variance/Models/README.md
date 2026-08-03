# Models

[Back to the package README](../README.md)

This directory contains the completed LightGBM tuning artifacts retained from
the variance experiment. It includes 15 logical tuning configurations, with a
separate P and Q model file and metadata record for each configuration.

These files are search snapshots, not one permanent pair of deployed
estimators. During 2022 rolling-origin evaluation, the retained parameters,
features, and preprocessing specification are read from metadata and fresh P
and Q estimators are refitted at every midnight origin.

## File naming

Publication artifacts use:

```text
S{origins}_T{trials}_seed{seed}_{target}_LGBM.txt
S{origins}_T{trials}_seed{seed}_{target}_LGBM.meta.json
```

For example, `S36_T60_seed17_Q_LGBM.txt` is the Q model retained after 60
terminal Optuna trials using sampler seed 17 and 36 chronological tuning
origins.

| Name part | Meaning |
|---|---|
| `S24`, `S36`, `S48` | Number of chronological origins used to score each candidate. |
| `T10`, `T30`, `T60` | Completed terminal-trial milestone; not tree count or forecast horizon. |
| `seed17`, `seed42`, `seed73` | Optuna TPE sampler seed. |
| `P`, `Q` | Active-power or reactive-power target. |
| `.txt` | Native LightGBM text-model artifact. |
| `.meta.json` | Features, parameters, preprocessing state, fitting boundary, objective, and provenance. |

## Retained design

| Tuning origins | Trial milestones | Seeds | Logical configurations |
|---:|---|---|---:|
| 24 | 10, 30, 60 | 17, 42, 73 | 9 |
| 36 | 60 | 17, 42, 73 | 3 |
| 48 | 60 | 17, 42, 73 | 3 |

All configurations use the same forecasting structure:

- separate P and Q LightGBM estimators;
- one shared hyperparameter vector for the pair;
- mutual P-Q history;
- explicit lags of 1 and 24 hours;
- 24- and 168-hour trailing means and sample standard deviations;
- synchronized 24-hour recursive prediction.

The seed-42 artifacts supply the five frozen configurations used in the 2022
forecast comparison. Seeds 17 and 73 are development-only replications used to
describe variation in the Optuna search path. LightGBM's fitting seed remains
fixed, so these replications vary the sampler path rather than tree-fitting
randomness.

## Historical anchor

`HIST_MUT_L24_P_LGBM.meta.json` and `HIST_MUT_L24_Q_LGBM.meta.json` describe
the historical mutual-history LightGBM anchor. The source experiment retained
metadata rather than publication copies of its fitted model files. When the
anchor is evaluated, fresh estimators are fitted from that stored
specification under the same rolling-origin protocol.

## Duplicate artifacts

Eight of the 30 target-model files are byte-identical to an earlier target
artifact. This occurs when the same best trial remains selected at a later
milestone or under another recorded design. They are kept because the labels
represent different completed search checkpoints. `duplicate_model_of` in the
manifests identifies every duplicate; it is not evidence of a second fit.

## Integrity manifests

- `MODEL_MANIFEST.json` is the authoritative machine-readable mapping between
  original snapshots and publication names.
- `MODEL_MANIFEST.csv` provides a compact tabular view.

Both include SHA-256 hashes, source paths, publication identifiers, target,
seed, origin count, trial milestone, and duplicate mappings.

## Git LFS

The model directory is approximately 682 MiB, and eight individual model files
exceed 50 MiB. The package-level `.gitattributes` routes
`Models/*_LGBM.txt` through Git LFS.

Before adding or cloning these artifacts:

```bash
git lfs install
```

After cloning:

```bash
git lfs pull
```

If a `.txt` file contains a short Git LFS pointer instead of a LightGBM model,
run `git lfs pull` before evaluation.

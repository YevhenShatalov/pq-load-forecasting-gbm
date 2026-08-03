# Results

[Back to the package README](../README.md)

This is the compact reader-facing result package. It contains the data,
tables, complete figures, and audit records needed to inspect the reported
conclusions without rerunning Optuna or the 59-day rolling evaluation.

Large evaluation caches, trial-by-fold traces, duplicate plots, smoke tests,
and diagnostics are intentionally excluded. They can be regenerated below
`../_work` by the public programs in `../code`.

## Reading configuration names

Each learned system is identified as `Sxx-Tyy`:

- `S24`, `S36`, or `S48` is the number of chronological 2021 forecast origins
  used to evaluate each candidate during tuning.
- `T10`, `T30`, or `T60` is the completed Optuna trial milestone. It is not
  the number of trees and not the 24-hour forecast horizon.

For example, `S36-T60` is the best eligible specification retained after 60
trials when each candidate was scored on 36 chronological tuning origins.

## Evaluation intervals

- **Primary:** 54 midnight origins from 1 January through 23 February 2022.
- **Descriptive:** 5 midnight origins from 24 through 28 February 2022.
- **All forecasts:** 59 trajectories, 24 hourly forecasts per target and
  configuration.

The final five origins remain available for descriptive stress analysis but
are excluded from the primary metrics and paired statistical tests.

## Main result

`S36-T60` is the numerical leader under both primary paired criteria:

| Criterion | `S36-T60` value |
|---|---:|
| P RMSE | 118.04 kW |
| Q RMSE | 115.24 kVAr |
| Fixed-scale paired RMSE | 0.5581 |
| Paired normalized CRPS | 0.2942 |

`S48-T60` records the lowest Q RMSE, 115.13 kVAr, by a small margin. The three
T60 breadth systems are not statistically separated after Holm correction.
Within the nested 24-origin depth trajectory, T60 has significantly lower
paired CRPS than T10 and T30, while point-loss differences do not remain
significant after adjustment.

Start with:

- [`tables/Table2_primary_performance.csv`](tables/Table2_primary_performance.csv)
  for the primary accuracy values;
- [`tables/Table3_paired_tests.csv`](tables/Table3_paired_tests.csv) for the
  adjusted pairwise decisions; and
- [`figures/06_paired_comparisons.png`](figures/06_paired_comparisons.png) for
  the visual summary of mean daily loss differences.

## `figures`

Every figure is supplied as PNG for quick viewing and as vector PDF for
publication.

| Prefix | Contents |
|---|---|
| `01_validation_origins` | Nested 24-, 36-, and 48-origin tuning schedules through 2021. |
| `02_search_depth_and_breadth` | Best-so-far depth trajectory and three-seed breadth rescoring on the common 48-origin schedule. |
| `03_forecast_trajectories` | P and Q forecasts over all 59 daily origins, with the final five days separated descriptively. |
| `04_point_accuracy` | P-versus-Q RMSE and a close view of the learned T60 systems. |
| `05_probabilistic_accuracy` | Active- and reactive-power CRPS. |
| `06_paired_comparisons` | Daily paired loss differences and 95% confidence intervals. |

For Figure 6, a negative difference favors the first named system. Green
intervals denote Holm-adjusted significance at the 5% family-wise level.

## `tables`

| File | Contents |
|---|---|
| `Table1_search_design_cost.csv` | Tuning origins, trial milestones, target-fold fits, and measured search time. |
| `Table2_primary_performance.csv` | Primary P/Q RMSE and CRPS plus the two fixed-scale paired scores. |
| `Table3_paired_tests.csv` | Daily paired contrasts, confidence intervals, raw p-values, Holm p-values, and decisions. |
| `breadth_three_seed_summary.csv` | Development-only seed-17, seed-42, and seed-73 breadth rescoring. |
| `full_results.xlsx` | Formatted workbook containing the principal evidence in one file. |

The CSV files are authoritative for programmatic use. The workbook is a
reader convenience and does not contain additional scientific evidence.

## `data`

| File | Unit of observation |
|---|---|
| `forecast_59days_long.csv` | One row per configuration, target, forecast timestamp, and horizon. |
| `forecast_59days_wide.csv` | The same forecasts arranged by timestamp for direct comparison. |
| `daily_paired_losses.csv` | One row per configuration and daily origin for paired point and CRPS tests. |
| `primary_54day_metrics.csv` | Detailed metrics over the primary interval. |
| `stress_5day_metrics.csv` | Descriptive metrics over 24-28 February only. |

Absolute P metrics are in kW and absolute Q metrics are in kVAr. Normalized
paired scores are dimensionless.

## `reproducibility`

| File | Purpose |
|---|---|
| `final_results_manifest.json` | SHA-256 inventory for the compact result package. |
| `finalization_report.md` | Record of the final package assembly. |
| `forecast_leakage_audit.json` | Structural and behavioral audit of all 59-origin forecasts. |
| `input_audit.json` | Validated data, split, scale, and historical-anchor checks. |
| `nested_split_audit.json` | Exact nesting checks for the 24-, 36-, and 48-origin designs. |
| `source_inventory.csv` | Source artifact inventory used to assemble the compact package. |
| `source_discrepancies.csv` | Any detected mismatch between expected and source artifacts. |

An empty `source_discrepancies.csv` data section indicates that no retained
source discrepancy was found.

## Statistical interpretation

Point metrics rank observed forecast errors. CRPS evaluates the complete
empirical predictive distribution formed from 92 fixed Q4 residual
trajectories. The paired tests then ask whether daily loss differences are
large relative to their time-dependent variation.

Tests use 54 complete daily trajectories, a Bartlett-Newey-West HAC variance
estimate with lag 3, a trajectory-level Harvey-Leybourne-Newbold correction,
two-sided p-values, and Holm correction within each three-comparison family.
A lower numerical metric does not automatically imply a statistically
separated system.

## Rebuild these results

From the package root, first generate the technical rolling forecasts and
probabilistic scores, then assemble the compact package:

```bash
python code/evaluate_models.py --plan final --execute
python code/compare_models.py --plan final --execute
```

To recreate `tables/full_results.xlsx`, add the optional workbook flag:

```bash
python code/compare_models.py --plan final \
  --build-results-workbook --execute
```

All public programs are preview-only unless `--execute` is supplied.

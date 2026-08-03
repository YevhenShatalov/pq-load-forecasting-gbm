# Input

[Back to the package README](../README.md)

This directory contains the authoritative prepared inputs used by the
LightGBM search-depth and validation-breadth experiment. File names are stable
because the public programs and manifests refer to them directly.

The prepared package can be evaluated without the original source workbooks.
Recreating `PQ.xlsx` from raw measurements requires six user-supplied files;
see [the code guide](../code/README.md#1-prepare-or-audit-data).

## Prepared hourly data

`PQ.xlsx` contains active power (`P_Power`, kW), reactive power (`Q_Power`,
kVAr), weather and calendar variables, target lags, and trailing statistics.

| Worksheet | Explicit target lags | Rows | First timestamp | Last timestamp |
|---|---|---:|---|---|
| `1` | 1 h | 13,102 | 2021-01-01 02:00 | 2022-06-30 23:00 |
| `24` | 1 and 24 h | 13,080 | 2021-01-02 00:00 | 2022-06-30 23:00 |
| `168` | 1, 24, and 168 h | 12,936 | 2021-01-08 00:00 | 2022-06-30 23:00 |

The reported experiment uses worksheet `24`. Each target model receives both
P and Q history, explicit 1- and 24-hour lags, and trailing mean and sample
standard deviation over 24 and 168 hours.

The workbook extends beyond every individual training cutoff. This does not
give an estimator access to future target values. During rolling evaluation,
future P and Q values are masked and all target-dependent inputs are rebuilt
from observations available at the origin and earlier recursive forecasts.

## Chronological split designs

| File | Role | Rows | Main origin range |
|---|---|---:|---|
| `splits_search24.xlsx` | Base tuning design | 24 | 2021-02-03 to 2021-09-27 |
| `splits_search36.xlsx` | Nested breadth extension | 36 | 2021-02-03 to 2021-09-30 |
| `splits_search48.xlsx` | Second nested breadth extension | 48 | 2021-02-03 to 2021-09-30 |
| `splits_gap48.xlsx` | Estimator-staleness sensitivity | 48 | 2021-02-16 to 2021-09-27 |
| `splits_event16.xlsx` | Later-event challenge, excluded from Optuna | 16 | 2021-10-13 to 2022-01-14 |
| `splits_historical.xlsx` | Original candidate-screening provenance | 123/123/122 | February to December 2021 |

The primary 24-origin design contains 12 regular, 6 calendar, and 6 operating
stress origins. The 36-origin design retains all 24 rows and adds 12 origins;
the 48-origin design retains all 36 rows and adds another 12. The extensions
preserve the same 2:1:1 regular/calendar/stress composition.

Each SEARCH-24, SEARCH-36, and SEARCH-48 row:

1. expands training from the fixed January start through 23:00 immediately
   before the forecast origin;
2. has `fit_gap_hours = 0`;
3. hides the next calendar day; and
4. scores one synchronized recursive 24-hour P-Q forecast.

There is no one-week exclusion interval in these three variance-experiment
designs. The 168-hour quantity used by the model is a trailing-statistic
window, not a validation gap and not an additional forecast horizon.

`splits_gap48.xlsx` is a separate sensitivity design. It crosses 12 origins
with fitting gaps of 0, 24, 72, and 168 hours while target history remains
available through the hour before the forecast day. It measures estimator
staleness and is not used in the main search-depth or breadth comparison.

`split_design_manifest.json` records the split-selection rules, strata, row
order, history policy, nesting checks, and intended use of every design.

## Supporting definitions

| File | Meaning |
|---|---|
| `objective_scales.json` | Frozen P and Q normalization scales calculated from the previous-day seasonal-naive benchmark over SEARCH-24. |
| `historical_mut_l24_params.json` | Verified parameter vector and provenance for the historical mutual-history, 1- and 24-hour-lag anchor. |
| `historical_source_mut_l24_P.meta.json` | Original active-power anchor metadata. |
| `historical_source_mut_l24_Q.meta.json` | Original reactive-power anchor metadata. |
| `device_selection.json` | CPU/GPU preflight and selected-device record. |

The fixed objective scales are 192.005583 kW for P and 229.797998 kVAr for Q.
They put both targets on a common dimensionless scale during Optuna tuning and
later paired scoring.

## Preparation notes

The data-preparation workflow merges P, Q, weather, daylight, calendar, and
holiday information; flags target outliers; fills resulting target gaps with
SARIMA-based estimates; and creates the lagged and trailing features. This
preparation is completed before the rolling experiment. Forecasting code still
masks future targets at every origin.

Future weather rows are treated as available over the 24-hour horizon. The
reported forecast accuracy is therefore conditional on those supplied
exogenous values.

## Integrity manifests

- `INPUT_MANIFEST.json` is the authoritative machine-readable inventory.
- `INPUT_MANIFEST.csv` provides the same mapping in tabular form.

The manifests include SHA-256 hashes, byte sizes, source provenance,
plain-language roles, workbook sheet dimensions, and date ranges. Validate the
complete input set with:

```bash
python code/prepare_data.py --action audit --execute
```

Run the command from the package root.


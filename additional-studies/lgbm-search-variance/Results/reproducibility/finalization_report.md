# Conference Finalization Report

Generated: 2026-08-01T13:29:13.385192+00:00

## Frozen scope

- No new Optuna tuning was performed.
- Five unique seed-42 learned configurations were retained.
- Three sampler seeds were used only for descriptive development rescoring.
- The primary period contains 54 origins (1 January-23 February 2022).
- The descriptive stress period contains five origins (24-28 February 2022).

## Validation

- Forecast leakage audit: **PASS**.
- Maximum future-target perturbation difference: 0.0.
- Maximum target-order difference: 0.0.
- All seven reported configurations contain 1,416 hourly forecasts per target.
- Actual target values agree across configurations and sources.
- Breadth squared-error p-values reproduce the archived values to 1e-12.

## Numerical interpretation

- S36-T60 is the numerical primary leader under fixed paired RMSE and paired CRPS.
- S48-T60 has the lowest Q RMSE among the five learned configurations.
- No search-depth squared-error or tuning-breadth contrast is significant after its three-test Holm adjustment.
- In the separate search-depth CRPS family, S24-T60 is lower than S24-T10 and S24-T30 after Holm adjustment (adjusted p = 0.02775 for both).
- The three-seed breadth trajectories are descriptive and do not establish a universal optimum.

# `src/` overview

This folder contains the main pipeline for:

1. **Data preparation & feature engineering** → `data_prep_and_features.py`  
2. **Model training & hyperparameter tuning** → `train_models.py`  
3. **Walk-forward evaluation & probabilistic scoring** → `evaluate_models.py`  
4. **Diebold–Mariano comparisons across models** → `dm_across_models.py`

---

## 1. `data_prep_and_features.py`

Builds the processed feature tables (`data/PQ.xlsx`) and predefined walk-forward splits (`data/splits.xlsx`) from raw load, weather, and sunrise/sunset data.

### 1.1 Time handling

- `_is_tzaware_dtype(dtype)`  
  Tiny helper: returns `True` if a pandas dtype is timezone-aware.

- `localize_fixed_utc2(df, ts_col="Datetime", ...)`  
  Parse `ts_col` and attach a **fixed UTC+2** tz (`Etc/GMT-2`) without shifting clock times.

- `convert_fixed_utc2_to_kyiv(df, ts_col="Datetime")`  
  Convert a fixed-UTC+2 datetime column to **Kyiv civil time** (`Europe/Kyiv`, with DST). Raises if the column is naive.

### 1.2 Calendar & holiday features

- `build_calendar_features(idx)`  
  On an index interpreted in Kyiv time, creates:
  - `hour`, `weekday`, `dayofyear`, etc.
  - Cyclic encodings: `hour_sin/cos`, `day_of_week_sin/cos`, `day_of_year_sin/cos`.

- `build_holiday_features(idx)`  
  Uses the `holidays` package for Ukraine to build:
  - `is_holiday`, `is_day_before_hol`, `is_day_after_hol`, `is_weekend`
  - Special days: `is_new_year`, `is_jan2`, `is_old_new_year`, `is_orthxmas`, `is_dec25`
  - `post_ny_decay`: smooth factor capturing “post-New-Year” normalization.

### 1.3 Solar / daylight & seasons

- `solar_features(srss_df)`  
  From a daily sunrise/sunset table, builds an **hourly** DataFrame (fixed UTC+2) with:
  - `Datetime`
  - Cyclic calendar features (hour/day-of-week/day-of-year)
  - `daylight_intensity` ∈ [0,1] ≈ sin of daylight phase (0 at night, ≈1 near solar noon).

- `add_season_features(df, datetime_col=None, hemisphere="north")`  
  Adds **meteorological season** features:
  - `season_idx` (0–3), `season_sin/cos`
  - `day_in_season`, `season_len`
  - `season_phase_sin/cos` (position within the season on a unit circle).

### 1.4 Lag & rolling features + splits

- `make_lag_features(df, targets=None, ..., lags=(1,24,168), drop_original=False)`  
  For each target (e.g. `P_Power`, `Q_Power`), creates:
  - Lagged values: `*_lag_1`, `*_lag_24`, `*_lag_168`
  - Rolling stats (on 1-step-shifted series, to avoid leakage):
    - `*_rmean_24`, `*_rstd_24`, `*_rmean_168`, `*_rstd_168`.

- `rolling_dates(data, max_months=12, test_offset_days=7, test_days=1)`  
  Builds a list of **walk-forward train/test splits** (dicts with `train_start/end`, `test_start/end`) under several schemes:
  - New Year “anchor” split  
  - `cumulative_months`  
  - `fixed_1m_step7d`  
  - `cum_m_plus_days` variants (+1…+6 days).

### 1.5 Outliers & SARIMA filling

- `nan_outliers(s)`  
  Uses MSTL (daily + weekly periods) plus simple thresholds to flag outliers in a series (large residuals, extreme lows, strong negative spikes) and set them to `NaN`.

- `fill_nan_with_sarima(df, time_col, value_col, ...)`  
  Fits a SARIMAX model (daily seasonality) and:
  - Treats NaNs and zeros as missing.
  - Returns:
    - Original series (`value_col`)
    - `*_filled` – gaps replaced by SARIMA predictions
    - `*_predicted` – full in-sample prediction.

### 1.6 Script entrypoint

When run as a script, `data_prep_and_features.py`:

1. Reads raw **P/Q load**, **weather**, and **sunrise/sunset** Excel files.
2. Merges them on hourly `Datetime` (fixed UTC+2).
3. Adds solar, season, calendar, and holiday features.
4. Detects/removes outliers in `P_Power` and `Q_Power`, fills with SARIMA.
5. For lag sets `[1]`, `[1,24]`, `[1,24,168]`:
   - Builds lag/rolling features and writes each version to a sheet in `data/PQ.xlsx`.
   - Builds matching walk-forward splits and writes them to `data/splits.xlsx`.

---

## 2. `train_models.py`

Handles **model selection and training** for the P/Q targets using gradient boosting models and Optuna:

- LightGBM, XGBoost, CatBoost  
- scikit-learn HistGradientBoostingRegressor (`HGBR`) and GradientBoostingRegressor (`GBR`)

Key ideas:

- **Leakage-safe lag handling** via `lag_policy` (`"drop"`, `"own"`, `"mutual"`).  
- **Feature selection** per fold (MI or SFS) on the train window only.  
- **Joint P & Q tuning** with optional mutual lags and peer predictions.  
- Proper handling of **categorical features** and model-specific preprocessing.  
- Saving best models in a **native format** plus JSON metadata.

### 2.1 Utilities & metrics

- Helpers for time differences, ensuring non-empty feature matrices, and simple NaN-safe median filling for feature selection.
- `evaluate_predictions(...)` computing:
  - `MAE`, `RMSE`, `R2`
  - `MAPE%`, `SMAPE%`, `WMAPE%`
  - `MASE` (with configurable seasonality).

### 2.2 Lag metadata & feature selection

- `infer_lag_meta(...)`  
  Parse feature names (`*_lag_*`, `*_rmean_*`, `*_rstd_*`) and infer owner target, lag length, rolling window, etc.

- `_build_sfs_estimator(...)`  
  Build light estimators for Sequential Feature Selection (ridge, HGBR, GBR, LGBM, XGB, CatBoost).

- `select_columns(selector, Xtr, ytr, ...)`  
  Leakage-safe feature selection on train only:
  - `"all"`: no selection.  
  - `"mi_top_k"`: mutual information with one-hot encoding of categoricals.  
  - `"sfs*"`: forward/backward SFS with `TimeSeriesSplit` and RMSE scorer.

### 2.3 Lag policy & rolling recursion

- `cv_pairs_from_splits(...)`  
  Turn `splits.xlsx` into `(train_idx, valid_idx)` arrays.

- `choose_features_for_split(...)`  
  For a given split/target, decide which lag/rolling features are:
  - **Safe long** (can be used directly), and  
  - Need to be **recomputed** at prediction time from observed/predicted histories, depending on:
    - Gap between train_end and test_start,  
    - Test horizon length,  
    - `lag_policy` (`"drop"`, `"own"`, `"mutual"`).

- `_y_or_pred`, `_rolling`  
  Helpers that mix observed values (inside train) and predicted values (inside test) when reconstructing short lags and rolling statistics.

### 2.4 Categorical handling & model-safe preprocessing

- `classify_features(...)`  
  Split non-target columns into `num_cols` / `cat_cols` via simple heuristics and a user-provided list of known categoricals.

- `prepare_X_for_model(model_name, X, categorical_cols, xgb_native_categorical=True)`  
  Normalizes feature dtypes and encoding per model:
  - Native categoricals for LGBM/XGB (if desired).
  - String/object categoricals for CatBoost.
  - One-hot encoding for HGBR/GBR or XGB without native cats.

- `freeze_categories`, `align_to_columns`, `safe_for_model`, `ensure_named_df`  
  Ensure consistent category levels and feature ordering across train/val/test and handle NaNs for models that require imputation.

### 2.5 Rolling prediction & validation matrices

- `roll_predict_multi(...)`  
  Multi-target 1-step recursion over the test horizon, updating short lags/rollings using observed + predicted values for all targets.

- `build_validation_matrices(...)`  
  For given CV folds, rebuild validation design matrices with the same lag policy and preprocessing as used in training.

### 2.6 Model definitions & Optuna objectives

- `default_params(model_name)` and `suggest_params(trial, model_name, ...)`  
  Baselines and search spaces for LightGBM, XGBoost, CatBoost, HGBR, and GBR.

- `build_estimator(model_name, params, use_gpu=True)`  
  Instantiate a model with appropriate GPU/CPU options.

- `_cv_scores_per_target_fixed(...)`  
  Compute per-target CV scores (RMSE, MAPE%) for a fixed set of models/features/lag policies.

- `_make_objective_shared(...)` / `_make_objective_single_target_given_peer(...)`  
  Optuna objectives for:
  - Shared hyperparameters across targets, or  
  - Alternating two-target tuning where one target uses the current best peer model for mutual lags.

### 2.7 Top-level tuner & persistence

- `tune_multi_targets(...)`  
  Main tuner:
  - Reads splits and lag metadata,  
  - Runs Optuna studies (shared or separate),  
  - Refits final models on the full dataset,  
  - Returns models + metadata (features, lag policy, CV metrics).

- `save_native(model, path, meta=None)` / `load_native(path)`  
  Save/load models in native formats plus JSON metadata.

- `_read_pq_xlsx(...)`, `_read_splits_xlsx(...)`  
  Helpers to load `PQ.xlsx` and `splits.xlsx`.

### 2.8 Script entrypoint

Example usage in `__main__`:

- Loop over PQ sheets and matching split sheets.  
- For each, run `tune_multi_targets(...)` with chosen:
  - model family (e.g. LGBM / CatBoost),  
  - feature selector,  
  - lag policy (`"mutual"` for P/Q recursion),  
  - hyperparameter sharing scheme.  
- Save tuned models and metadata into a models directory.

---

## 3. `evaluate_models.py`

This module takes the **tuned models** (`*_best.meta.json` + native files) and the processed `PQ.xlsx` and produces **out-of-sample evaluation workbooks** with:

- Per-timestamp predictions, prediction intervals, quantiles, CRPS, and pinball losses.  
- Per-day metrics (RMSE, MAE, MAPE%, SMAPE%, WMAPE%, MASE, CRPS, mean pinball over τ).

The main output is a set of Excel files under `SAVE_DIR` (e.g. `Evaluated/`):

- `P_Power_predictions`, `P_Power_daily_metrics`  
- `Q_Power_predictions`, `Q_Power_daily_metrics`  

for each model family / lag policy combination.

### 3.1 Config & path handling

Top-level config controls:

- `DATA_PATH` – path to `PQ.xlsx`.  
- `MODELS_DIR` – folder with `*_best.*` and `*_best.meta.json`.  
- `SAVE_DIR` – where the evaluation workbooks are saved.  
- Evaluation window: `TRAIN_START`, `TRAIN_END`, `TEST_START`, `TEST_END`.  
- Walk-forward mode: expanding vs. sliding window, daily horizons (`DAY_AHEAD_HOURS`).  
- Probabilistic settings:  
  - `INTERVAL_LEVELS`, `ALL_INTERVAL_COVERAGES`, `QUANTILE_GRID`  
  - `N_SIM_PI` (number of simulated paths)  
  - `BLOCK_SIZE_PI` (for bootstrap when needed).

`resolve_paths()` allows environment variables and automatically falls back to `/mnt/data` in notebook environments.

### 3.2 Metrics and probabilistic scores

- Classical metrics: `rmse`, `mae`, `mape`, `smape`, `wmape`, `mase`, wrapped in `evaluate_predictions(...)` (which also uses training data for MASE scaling).  
- Probabilistic scores:
  - `pinball_loss(y, q, tau)` – per-forecast pinball loss.  
  - `crps_from_sims(y, sims)` – CRPS from simulated residual paths.

### 3.3 Lag/rolling recomputation & feature classification

- `parse_feature(name)` / `recompute_feature_at(ts, name, yhist)`  
  Understand lag/rolling feature names (`*_lag_*`, `*_rmean_*`, `*_rstd_*`) and recompute them at timestamp `ts` using a mixed history of observed + predicted values (same idea as in training).

- `iter_day_blocks_for_sheet(...)`  
  Yield **day-by-day** train/test blocks (24-hour day-ahead horizons) clipped to each PQ sheet’s index and respecting expanding or sliding windows.

- `identify_targets(df)`  
  Find P and Q column names (`P_Power/Q_Power` or `P/Q`).

- `classify_features(df, KNOWN_CATS)`  
  Split features into numerical and categorical, using a list of known categoricals and dtype/low-cardinality heuristics.

### 3.4 Model reconstruction & artifact parsing

- `read_pq_xlsx(DATA_PATH)`  
  Read all PQ sheets, reconstruct a `DatetimeIndex` for each, floor to hourly, and sort.

- `read_artifacts(MODELS_DIR)`  
  Parse filenames like  
  `lagdrop_LGBM_sheet1_P_Power_best.meta.json`  
  into `Artifact` objects containing:
  - model name, lag policy, sheet index, target,  
  - tuned params, feature list, recalc feature list,  
  - path to the native model file.

- `recover_params_from_model_file(art)`  
  Fallback: read params back from saved LGBM/XGB native models if the meta JSON doesn’t contain them.

- `build_estimator_from_params(...)`  
  Recreate LightGBM/XGBoost/CatBoost/HGBR/GBR estimators with the saved parameters (GPU/CPU safe).

### 3.5 Prediction modes (drop / own / mutual)

- `roll_predict_joint(...)`  
  Joint 1-step-ahead recursion for P & Q using **mutual lag policy**: both targets are predicted at each hour, then used to reconstruct short lags/rollings at the next hour.

- `roll_predict_own_single(...)`  
  1-step recursion for a single target under **own-lags** policy (no mutual recursion).

- `direct_predict(...)`  
  No recursion: build full test matrix, predict in one shot (used for `lag_policy="drop"`).

These functions apply the same feature preprocessing as in training (`prepare_X_for_model`, `freeze_categories`, `align_to_columns`, `safe_for_model`).

### 3.6 Prediction intervals, quantiles, CRPS, pinball

- `moving_block_bootstrap_indices(...)`  
  Generate block bootstrap indices for residual resampling.

- `residual_block_pi(...)`  
  Residual-based prediction intervals:
  - Either **hour-matched** resampling (residual pool per hour of day), or  
  - Moving block bootstrap over residuals.  
  - Returns equal-tailed intervals for requested coverages and the full matrix of simulated paths (`sims`).

- `_add_quantiles_intervals_scores_to_df(base_df, sims, y_true_vec, ...)`  
  Given simulated paths and base prediction DataFrame, append:
  - Quantiles `Q05..Q95` (for `QUANTILE_GRID`).  
  - Full set of intervals `Lxx/Uxx` for `ALL_INTERVAL_COVERAGES`.  
  - Per-timestamp CRPS.  
  - Per-timestamp pinball losses for all τ in `QUANTILE_GRID`.

### 3.7 Single-split vs. walk-forward experiments

All results are stored in an `ExperimentResult` dataclass with:

- Tag (`exp`), model name, lag policy, target, other target.  
- `y_true`, `y_pred`, time index, aggregated metrics.  
- `pred_df` – per-timestamp predictions + intervals + scores.  
- `daily_metrics` – per-day metrics (one row per day).

Two main evaluation modes:

- `fit_refit_and_eval_sheet(...)`  
  - Single train/test split for a sheet (train once, test over a range).  
  - Useful for fixed windows.

- `fit_refit_and_eval_sheet_walkforward(...)`  
  - **Daily walk-forward refits**:
    - For each day in `[TEST_START, TEST_END]`, build a new train window (expanding or sliding).  
    - Refit P and Q models, predict 24 hours ahead, simulate residuals, compute metrics.  
  - Accumulates per-day predictions/metrics and then aggregates into a single `ExperimentResult` per target.

### 3.8 Grouping artifacts & saving results

- `group_artifacts_by_family(arts)`  
  Group tuned P/Q models by `(family_token, model_name)` where `family_token` roughly captures lag policy / selection strategy (e.g. `mutual_lags_LGBM`).

- `choose_best_pq_sheet_for_pair(...)`  
  For a (P,Q) artifact pair, pick the PQ sheet that:
  - Contains all required features for both targets, and  
  - Maximizes the available training rows in the requested window.

- `run_experiment_for_pair(...)`  
  For a family + (P,Q) pair:
  - Choose PQ sheet, resolve split for this sheet.  
  - Run either single-split or walk-forward evaluation.  
  - Return all `ExperimentResult`s and an Excel filename stub.

- `save_results_for_family(...)`  
  Write evaluation outputs per family to Excel (or CSV fallback) under `SAVE_DIR`:
  - For each target:
    - `<target>_predictions` – stacked `pred_df` for all experiments.  
    - `<target>_daily_metrics` – stacked per-day metrics for all experiments.

### 3.9 Script entrypoint

`main()` wires everything:

1. Resolve paths (`DATA_PATH`, `MODELS_DIR`) and load all PQ sheets.  
2. Read tuned artifacts from `MODELS_DIR` and recover missing params if needed.  
3. Group artifacts by family/model; keep only sheet pairs that have both P and Q.  
4. For each family and sheet pair:
   - Run experiments (`run_experiment_for_pair`).  
   - Save evaluation workbooks (`save_results_for_family`).

---

## 4. `dm_across_models.py`

This module operates **after** `evaluate_models_walkforward.py`. It reads the Excel workbooks in `Evaluated/` and performs:

- **Pairwise Diebold–Mariano tests** across models for each target.  
- A set of **diagnostic plots**:
  - Per-model forecast plots with 80/95% intervals.  
  - Per-model daily pinball/CRPS trend plots.  
  - Global daily pinball/CRPS overlays across models (for P and Q).  
  - Global daily overlays of classical error metrics (RMSE, MAE, MAPE%, SMAPE%, WMAPE%, MASE).

Outputs go into:

- `Evaluated/DM/DM_across_models.xlsx` – DM statistic / p-value / sample size matrices.  
- `Evaluated/DM/plots/` – PNG plots for each model and global overlays.

### 4.1 Input normalization

- `read_model_file(path)`  
  For each evaluation workbook, pick relevant sheets for `P_Power` and `Q_Power`, normalize to a standard schema with columns:
  - `ts`, `date`, `y_true`, `y_pred`  
  plus optional `L80/U80`, `L95/U95`, quantiles `Qxx`, pinball columns `Pinball_ττ`, and `CRPS`.

- `_normalize_pred_df(...)`, `_pick_sheet(...)`  
  Handle different sheet names / column naming conventions and coerce types.

### 4.2 Pinball, quantiles, CRPS reconstruction

- `_pinball_columns(df)` / `_quantile_columns(df)` / `_tau_from_quantile_name(name)`  
  Discover which columns hold per-τ pinball losses and quantile forecasts.

- `_pinball_from_quantiles_row(...)` / `_crps_from_quantiles_row(...)`  
  For a single timestamp, compute:
  - Average pinball across the available quantiles.  
  - Discrete CRPS approximation from quantile losses (2× integral of pinball over τ).

These are used when explicit pinball/CRPS columns are missing but quantiles are present.

### 4.3 Per-model plots

- `plot_forecast_with_intervals(df, title, out_png)`  
  Time series plot of `y_true` vs `y_pred` plus 80/95% bands (`L80/U80`, `L95/U95`), with a visibility gate so extremely narrow intervals are not overplotted.

- `plot_daily_pinball_and_crps(df, title, out_png)`  
  Build daily mean pinball / CRPS series and plot them over time for a single model.

Both use `_decorate_axes` and `_ensure_dir` for neat, consistent formatting.

### 4.4 Global daily trend plots (Pinball / CRPS)

- `_compute_daily_scores(df)`  
  From per-timestamp predictions, produce a daily table (`date`, `Pinball_mean`, `CRPS`).

- `_parse_family_variant(model_id)`  
  Parse a model ID into a `(family, variant)` pair (e.g. family = `"mutual_lags_LGBM"`, variant = `"sheet2"`). Used to define consistent styles across P and Q.

- `_family_color_map(families)` / `_build_style_map(items)`  
  Assign:
  - **Color per family** (using `tab20`),  
  - **Dash style per variant** within a family, so global plots remain readable even with many models.

- `build_global_daily_trend_plots(models_by_target, out_dir)`  
  For P and Q separately, build:
  - `global_daily_Pinball_P.png`, `global_daily_CRPS_P.png`  
  - `global_daily_Pinball_Q.png`, `global_daily_CRPS_Q.png`  
  Each line = one model; color encodes family, dash style encodes variant.

### 4.5 Global daily error metric plots

- `_compute_daily_error_metrics(df)`  
  For a model’s time series, compute daily:
  - `RMSE`, `MAE`, `MAPE%`, `SMAPE%`, `WMAPE%`, `MASE`.

- `build_global_daily_metric_plots(models_by_target, out_dir)`  
  For P and Q, build overlays for each metric:
  - `global_daily_RMSE_P.png`, `global_daily_MAE_P.png`, …  
  - `global_daily_RMSE_Q.png`, `global_daily_MAE_Q.png`, …  
  Again: color by family, dash style by variant.

### 4.6 Diebold–Mariano matrices

- `_pairwise_align_predictions(A, B)`  
  Align two models by timestamp, keep a single `y` series, form `p_i` and `p_j` predictions.

- `dm_pairwise_matrix(models, h=H)`  
  For all model pairs for a target, run Diebold–Mariano tests with **squared-error loss** and horizon `H` (e.g. 24 for day-ahead):

  - DM statistic matrix  
  - p-value matrix  
  - overlap count matrix (`n`).

These matrices are written to sheets:

- `<target>_DM_stat`  
- `<target>_DM_p`  
- `<target>_DM_n`  

inside `DM_across_models.xlsx`.

### 4.7 Orchestration

- `load_all_models(INPUT_DIR)`  
  Read all `.xlsx` files in `INPUT_DIR`, normalize them, and build a dict:
  - `{"P_Power": {model_id: df}, "Q_Power": {model_id: df}}`  
  plus a metadata table (`models_meta.csv`) with file/model IDs and chosen experiment variant (if multiple `exp` values exist).

- `main()`  

  1. Check `INPUT_DIR`, create `OUTPUT_DIR`, subfolder `plots/`.  
  2. Load all models and save metadata.  
  3. For each target/model:
     - Plot forecast + intervals (`forecast_*.png`).  
     - Plot daily pinball & CRPS (`scores_*.png`).  
  4. Build global daily Pinball/CRPS and classical metric plots for P/Q.  
  5. Compute DM matrices for each target and write them into `DM_across_models.xlsx`.

---


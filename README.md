# pq-load-forecasting-gbm
Code and experiments for probabilistic P/Q load forecasting with gradient boosting models (LightGBM, XGBoost, CatBoost, etc.). Includes leakage-safe lag construction, hyperparameter tuning, walk-forward evaluation, and Diebold–Mariano model comparison.

The dataset is derived from measurements at a **110/10 kV substation located in Kyiv oblast (Ukraine)**, aggregated to hourly active (P) and reactive (Q) power.
## Overview

This repository accompanies a research study on day-ahead probabilistic load forecasting of active (P) and reactive (Q) power. It implements:

- **Leakage-safe lag construction** across multiple lag policies (own, mutual, shared-mutual, lag-drop)
- **Calendar, seasonal, solar/daylight, and holiday features**
- **Feature selection** (all features, mutual information top-k, and sequential forward selection)
- **Hyperparameter tuning** with Optuna
- **Walk-forward evaluation** with proper multi-step forecasting
- **Probabilistic metrics** (pinball loss, CRPS) and
- **Diebold–Mariano (DM) comparisons** across models

The goal is to systematically train many GBM-type models and select the best-performing configurations for P and Q.

## Repository structure

```text
pq-load-forecasting-gbm/
  README.md
  requirements.txt           # Python dependencies

  src/
    data_prep_and_features.py       # Data loading, cleaning, feature engineering, lag building
    train_models.py                 # Train/tune GBM models for P and Q
    evaluate_models_walkforward.py  # Walk-forward forecasting + probabilistic metrics
    dm_across_models.py             # DM tests and comparison plots across models

  data/
    PD_2021History_GMT2.xlsx      # Raw active power demand (P) for 2021
    PD_2022History_GMT2.xlsx      # Raw active power demand (P) for 2022
    QD_2021History_GMT2.xlsx      # Raw reactive power demand (Q) for 2021
    QD_2022History_GMT2.xlsx      # Raw reactive power demand (Q) for 2022
    Sunrise_sunset.xlsx           # Sunrise/sunset and daylight-related information
    PD_Weather_GMT2.xlsx          # Weather data aligned to GMT+2 (or dummy structure)
    PQ.xlsx                         # Merged and processed P/Q dataset used by the pipeline
    splits.xlsx                     # Predefined train/validation/test split dates
    README_data.md                  # Notes on how to obtain or reconstruct the data

  models/
    # Saved best models and metadata, e.g.:
    # <model_name>_best.pickle
    # <model_name>_best.meta.json

  evaluated/
    # Excel/CSV outputs from evaluate_models_walkforward.py
    # P_Power_predictions_*.xlsx
    # Q_Reactive_power_predictions_*.xlsx
    # *_daily_metrics_*.xlsx

  results/
    # Outputs from DM analysis:
    # DM_across_models_P.xlsx
    # DM_across_models_Q.xlsx
    # related figures

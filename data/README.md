# Data

This folder contains the raw and processed data used in the study on probabilistic P/Q day-ahead load forecasting with gradient boosting models.

All time series come from a **110/10 kV substation in Kyiv oblast (Ukraine)**. Active (P_Power) and reactive (Q_Power) power are aggregated to **hourly resolution** and aligned on a **fixed UTC+02:00** timeline. Timestamps are stored as naive `"YYYY-MM-DD HH:MM:SS"` strings in Excel but should be interpreted as UTC+2.

---

## 1. Raw load time series

### 1.1 Active power (P)

- `PD_2021History_GMT2.xlsx`  
- `PD_2022History_GMT2.xlsx`

Both files have the same structure:

- `Datetime`  
  - Hourly timestamps from:
    - `0_PD_2021History_GMT2.xlsx`: **2021-01-01 00:00** to **2021-12-31 23:00** (8760 rows).  
    - `0_PD_2022History_GMT2.xlsx`: **2022-01-01 00:00** to **2022-06-30 23:00** (4344 rows).  
  - Interpreted as fixed UTC+2.

- `P_Power`  
  - Hourly active power at the 110/10 kV bus.  
  - Units are as in the original SCADA export (no additional scaling is applied in this repo).

### 1.2 Reactive power (Q)

- `QD_2021History_GMT2.xlsx`  
- `QD_2022History_GMT2.xlsx`

Same structure as P, but for reactive power:

These four files are the **primary load measurements** from which the `P_Power` and `Q_Power` targets in `PQ.xlsx` are derived.

---

## 2. Weather data

- `PD_Weather_GMT2.xlsx`

**Columns**

- `Datetime`  
  - Hourly timestamps from **2021-01-01 00:00** to **2023-01-02 01:00** (17546 rows).  
  - The forecasting experiments only use the intersection with the load period (up to **2022-06-30 23:00**).

- `Temperature`  
- `DF_Point` – dew point temperature.  
- `Relative_Humidity`  
- `Precipitation`  
- `Surface_Pressure`  
- `Wind_Speed`  
- `Wind_Speed_50` – wind speed at 50 m (if provided by the weather source).  
- `UV_Index`  
- `Irradiance` – global horizontal irradiance (dummy values here).  
- `PAR_Total` – photosynthetically active radiation (dummy values here).  
- `Rainy` – binary indicator (`1` = “rainy hour”, `0` = otherwise), derived from precipitation.

All these columns are later merged into `PQ.xlsx` and used as exogenous features.

---

## 3. Sunrise/sunset and solar geometry

- `Sunrise_sunset.xlsx`

Daily sunrise/sunset data for the substation’s location, used to derive smooth daylight and seasonal features.

**Columns**

- `date`  
  - Date (no time component), from **2021-01-01** to **2022-06-30** (546 rows).

- `sunrise_str`  
  - Local sunrise time on that date (e.g., `07:56:58`).

- `sunset_str`  
  - Local sunset time (e.g., `16:05:22`).

- `day_length_str`  
  - Day length as a string (e.g., `"8h 8m 23s"`).

In the preprocessing pipeline these are:

1. Attached to the fixed UTC+2 hourly index.  
2. Converted to a **smooth daylight intensity profile** per hour.  
3. Used to build features like `daylight_intensity` and meteorological season indicators in `PQ.xlsx`.

---

## 4. Processed feature set: `PQ.xlsx`

- `PQ.xlsx`

This is the **main merged and feature-rich dataset** used for model training and evaluation. It combines:

- Cleaned and imputed load series (`P_Power`, `Q_Power`).  
- Weather features.  
- Solar/daylight features.  
- Calendar, seasonal, and holiday features.  
- Multiple lag and rolling-window features.

The file contains **three sheets**, corresponding to different lag configurations:

- Sheet `"1"` – only 1-hour lags.
- Sheet `"24"` – 1-hour and 24-hour lags.
- Sheet `"168"` – 1-hour, 24-hour, and 168-hour (weekly) lags.

### 4.1 Coverage and size

- `"1"`  
  - 13 102 rows × 51 columns.  
  - Coverage: **2021-01-01 02:00** to **2022-06-30 23:00**.  
  - Early hours are dropped due to lag/rolling windows.

- `"24"`  
  - 13 080 rows × 53 columns.  
  - Coverage: **2021-01-02 00:00** to **2022-06-30 23:00**.

- `"168"`  
  - 12 936 rows × 55 columns.  
  - Coverage: **2021-01-08 00:00** to **2022-06-30 23:00**.

All timestamps are hourly and should again be interpreted as fixed UTC+2.

### 4.2 Column groups

Below is the full column set for sheet `"1"`. Sheets `"24"` and `"168"` add extra lag columns as noted.

#### Targets

- `P_Power` – active power (cleaned/imputed).  
- `Q_Power` – reactive power (cleaned/imputed).

#### Lagged values and rolling statistics

- `P_Power_lag_1`  
- `P_Power_rmean_24`, `P_Power_rstd_24` – rolling mean/std over 24 hours.  
- `P_Power_rmean_168`, `P_Power_rstd_168` – rolling mean/std over 168 hours.  

- `Q_Power_lag_1`  
- `Q_Power_rmean_24`, `Q_Power_rstd_24`  
- `Q_Power_rmean_168`, `Q_Power_rstd_168`

Additional lag columns in other sheets:

- Sheet `"24"`:  
  - `P_Power_lag_24`, `Q_Power_lag_24`.

- Sheet `"168"`:  
  - `P_Power_lag_24`, `P_Power_lag_168`  
  - `Q_Power_lag_24`, `Q_Power_lag_168`.

These lags are constructed in a **leakage-safe** way (respecting the time index and the split definitions).

#### Weather features (from `PD_Weather_GMT2_dummy.xlsx`)

- `Temperature`  
- `DF_Point`  
- `Relative_Humidity`  
- `Precipitation`  
- `Surface_Pressure`  
- `Wind_Speed`  
- `Wind_Speed_50`  
- `UV_Index`  
- `Irradiance`  
- `PAR_Total`  
- `Rainy` (0/1)

#### Calendar & cyclic time features

- `hour` – hour of day (0–23).  
- `hour_sin`, `hour_cos` – 24-hour cyclic encoding.  

- `weekday` – day of week (0=Monday,…,6=Sunday).  
- `day_of_week_sin`, `day_of_week_cos` – 7-day cyclic encoding.  

- `dayofyear` – day of year (1–365/366).  
- `day_of_year_sin`, `day_of_year_cos` – annual cyclic encoding.

All these are computed in **Kyiv civil time** (`Europe/Kyiv`, including DST) after localizing the underlying index to fixed UTC+2.

#### Solar / daylight and season features

Derived from `0_Sunrise_sunset.xlsx`:

- `daylight_intensity`  
  - Smooth function of solar elevation (≈0 at night, peaked around local solar noon), in [0, 1].

- `season_idx` – integer in {0, 1, 2, 3} (winter, spring, summer, autumn, using meteorological seasons).  
- `season_sin`, `season_cos` – cyclic encoding of the four seasons.  
- `day_in_season` – day index within the season.  
- `season_len` – length of the current season (days).  
- `season_phase_sin`, `season_phase_cos` – normalized position within the season, mapped to the unit circle.

#### Holiday and special-day features (Ukraine)

Based on the `holidays` package for Ukraine plus manually specified dates:

- `is_holiday` – official public holiday (per calendar).  
- `is_day_before_hol` – day immediately before a public holiday.  
- `is_day_after_hol` – day immediately after a public holiday.  
- `is_weekend` – Saturday/Sunday.  

Specific fixed dates:

- `is_new_year` – January 1.  
- `is_jan2` – January 2.  
- `is_old_new_year` – January 14 (Old New Year).  
- `is_orthxmas` – January 7 (Orthodox Christmas).  
- `is_dec25` – December 25 (Western Christmas).

Post-New-Year effect:

- `post_ny_decay` – smooth “decay factor” capturing gradual normalization of load in early January (non-zero only during the first ~two weeks of January).

---

## 5. Predefined train/validation/test splits: `splits.xlsx`

- `splits.xlsx`

Contains **precomputed walk-forward splits** that correspond exactly to the rows in each `PQ.xlsx` sheet.

Sheets:

- `"1"` – splits for `PQ.xlsx` sheet `"1"`.  
- `"24"` – splits for sheet `"24"`.  
- `"168"` – splits for sheet `"168"`.

Each sheet has one row per split with the following columns:

- `Unnamed: 0` – simple integer index of the split (can be safely dropped when loading).  

- `train_start` – start timestamp of the training window.  
- `train_end` – end timestamp of the training window (inclusive).  

- `test_start` – start date of the test block (the script then uses it to build 24-hour day-ahead horizons).  
- `test_end` – end timestamp of the test block (inclusive).

- `scheme` – textual label describing how the split was generated, e.g.:  
  - `cumulative_months` – training window grows cumulatively month-by-month.  
  - `cum_m_plus_days` – cumulative months plus extra days.  
  - `fixed_1m_step7d` (if present) – fixed 1-month window sliding by 7 days.

- `is_new_year_test` – `True` if the test block covers the New Year period (useful for analyzing that regime separately).  
- `extra_days` – number of additional days added to the cumulative-months window (used for `cum_m_plus_days`).  
- `shift_days` – integer shift of the training window (used for fixed-window schemes).

These splits are what the training/evaluation scripts use for **reproducible walk-forward backtesting** and **DM comparisons**.

---

## 6. Time axis and timezone handling

To avoid daylight-saving complications:

1. All series are first placed on a **fixed UTC+2** index (`Etc/GMT-2`).  
2. Calendar and holiday features are computed in **`Europe/Kyiv`** time (including DST) based on this index.  
3. Solar/daylight features are computed by combining the fixed UTC+2 hourly index with local sunrise/sunset times.  
4. Before writing to Excel, timezones are stripped; `Datetime` is stored as a naive string but should always be read back as UTC+2.

---


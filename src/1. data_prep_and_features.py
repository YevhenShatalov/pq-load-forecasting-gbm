from __future__ import annotations
import operator
from statsmodels.tsa.seasonal import MSTL
from typing import Iterable, Optional, cast, List, Dict, Union, Any, Optional, Sequence
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper

TZ_FIXED_UTC2 = "Etc/GMT-2"   # fixed UTC+02:00 (POSIX sign reversed)
TZ_KYIV       = "Europe/Kyiv" # Kyiv civil time (EET/EEST with DST)


def _is_tzaware_dtype(dtype) -> bool:
    return isinstance(dtype, pd.DatetimeTZDtype)

def localize_fixed_utc2(
    df: pd.DataFrame,
    ts_col: str = "Datetime",
    *, 
    dayfirst: bool = False,
    explicit_format: Optional[str] = None
) -> pd.DataFrame:

    # 1) Parse to naive datetimes
    dt = pd.to_datetime(
        df[ts_col],
        format=explicit_format,
        dayfirst=dayfirst,
        errors="coerce",
    )
    # 2) Attach a fixed UTC+02:00 zone (does not shift the clock values)
    df = df.copy()
    df[ts_col] = dt.dt.tz_localize(TZ_FIXED_UTC2)
    return df

def convert_fixed_utc2_to_kyiv(df, ts_col: str = "Datetime") -> pd.DataFrame:
    if not _is_tzaware_dtype(df[ts_col].dtype):
        raise TypeError(
            f"{ts_col} must be tz-aware before converting. "
            f"Call localize_fixed_utc2(...) first."
        )
    out = df.copy()
    out[ts_col] = out[ts_col].dt.tz_convert("Europe/Kyiv")
    return out

def build_calendar_features(idx):
    base_idx = idx if getattr(idx, 'tz', None) is not None else pd.DatetimeIndex(idx).tz_localize(TZ_FIXED_UTC2)
    s = base_idx.tz_convert(TZ_KYIV)
    df = pd.DataFrame(index=base_idx)
    df["hour"] = s.hour
    df["dow"] = s.dayofweek
    df["dom"] = s.day
    df["month"] = s.month
    df["doy"] = s.dayofyear
    df["is_weekend"] = df["dow"].isin([5,6]).astype(int)
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    df["dow_sin"] = np.sin(2*np.pi*df["dow"]/7)
    df["dow_cos"] = np.cos(2*np.pi*df["dow"]/7)
    df["doy_sin"] = np.sin(2*np.pi*df["doy"]/366.0)
    df["doy_cos"] = np.cos(2*np.pi*df["doy"]/366.0)
    return df

def build_holiday_features(idx):
    # Compute holiday flags in local Kyiv civil dates to be DST-safe.
    # Do NOT use row-based shifts (e.g., +/-24) because DST days are 23/25 hours.
    # Accept both tz-aware and naive indexes (naive assumed to be in fixed UTC+02).
    base_idx = idx if getattr(idx, 'tz', None) is not None else pd.DatetimeIndex(idx).tz_localize(TZ_FIXED_UTC2)
    dt_idx = pd.DatetimeIndex(base_idx.tz_convert(TZ_KYIV))
    dates = dt_idx.date
    df = pd.DataFrame(index=idx)
    years = sorted(set(dt_idx.year))
    ua = holidays.country_holidays("UA", years=years, observed=True)

    # same-day holiday indicator
    is_hol = np.fromiter(((d in ua) for d in dates), dtype=int)
    df["is_holiday"] = is_hol

    # Previous/next day relative to local date, independent of number of hours in the day
    next_is_holiday = np.fromiter((((d + timedelta(days=1)) in ua) for d in dates), dtype=int)
    prev_is_holiday = np.fromiter((((d - timedelta(days=1)) in ua) for d in dates), dtype=int)
    df["is_day_before_hol"] = next_is_holiday
    df["is_day_after_hol"]  = prev_is_holiday

    # weekend flag based on local Kyiv day-of-week (Sat=5, Sun=6)
    dow = pd.Series(dt_idx.dayofweek, index=idx)
    df["is_weekend"] = dow.isin([5, 6]).astype(int)

    mmdd = pd.Series([d.strftime("%m-%d") for d in dates], index=idx)
    df["is_new_year"] = (mmdd == "01-01").astype(int)
    df["is_jan2"] = (mmdd == "01-02").astype(int)
    df["is_old_new_year"] = (mmdd == "01-14").astype(int)
    df["is_orthxmas"] = (mmdd == "01-07").astype(int)
    df["is_dec25"] = (mmdd == "12-25").astype(int)
    doy = pd.Series(dt_idx.dayofyear, index=idx)
    is_jan = (doy <= 14)
    dsny = (doy - 1).clip(lower=0)
    decay = np.exp(-dsny/3.0)
    df["post_ny_decay"] = np.where(is_jan, decay, 0.0)
    return df

def solar_features(srss_df: pd.DataFrame) -> pd.DataFrame:
    if srss_df.shape[1] < 3:
        raise ValueError("solar_features expects at least three columns (date, sunrise, sunset).")

    df = srss_df.copy()

    col_date = df.columns[0]
    col_sunrise = df.columns[1]
    col_sunset = df.columns[2]
    col_daylen = df.columns[3] if df.shape[1] > 3 else None

    # Parse the base date and attach Kyiv civil sunrise/sunset times adjusted to the fixed UTC+02 timeline.
    df[col_date] = pd.to_datetime(df[col_date].astype(str).str.strip(), format="%Y-%m-%d", errors="coerce")
    if df[col_date].isna().any():
        raise ValueError("Failed to parse dates in solar data.")

    dt_str_sunrise = df[col_date].dt.strftime("%Y-%m-%d") + " " + df[col_sunrise].astype(str).str.strip()
    dt_str_sunset = df[col_date].dt.strftime("%Y-%m-%d") + " " + df[col_sunset].astype(str).str.strip()

    dt_sunrise = pd.to_datetime(dt_str_sunrise, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    dt_sunset = pd.to_datetime(dt_str_sunset, format="%Y-%m-%d %H:%M:%S", errors="coerce")

    local_sunrise = dt_sunrise.dt.tz_localize(TZ_KYIV, ambiguous="infer", nonexistent="shift_forward").dt.tz_convert(TZ_FIXED_UTC2)
    local_sunset = dt_sunset.dt.tz_localize(TZ_KYIV, ambiguous="infer", nonexistent="shift_forward").dt.tz_convert(TZ_FIXED_UTC2)

    time_fmt = "%H:%M:%S"
    df[col_sunrise] = pd.to_datetime(local_sunrise.dt.strftime(time_fmt), format=time_fmt)
    df[col_sunset] = pd.to_datetime(local_sunset.dt.strftime(time_fmt), format=time_fmt)

    if col_daylen is not None:
        df[col_daylen] = pd.to_datetime(df[col_daylen].astype(str).str.strip(), format="%Hh %Mm %Ss", errors="coerce")

    rename_map = {}
    if col_date != "date":
        rename_map[col_date] = "date"
    if col_sunrise != "sunrise":
        rename_map[col_sunrise] = "sunrise"
    if col_sunset != "sunset":
        rename_map[col_sunset] = "sunset"
    if col_daylen is not None and col_daylen != "day_length":
        rename_map[col_daylen] = "day_length"
    if rename_map:
        df = df.rename(columns=rename_map)

    date_col = "date"
    sunrise_col = "sunrise"
    sunset_col = "sunset"
    daylen_col = "day_length" if col_daylen is not None else None

    hours = pd.DataFrame({"hour": np.arange(24, dtype=int)})
    hourly = df.merge(hours, how="cross")
    hourly["Datetime"] = hourly[date_col] + pd.to_timedelta(hourly["hour"], unit="h")
    hourly = hourly.drop(columns=["hour"])
    hourly = hourly.sort_values("Datetime").reset_index(drop=True)

    out = hourly.drop(columns=[date_col])

    out["hour"] = out["Datetime"].dt.hour
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24).round(5)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24).round(5)

    out["weekday"] = out["Datetime"].dt.dayofweek
    out["day_of_week_sin"] = np.sin(2 * np.pi * out["weekday"] / 7).round(5)
    out["day_of_week_cos"] = np.cos(2 * np.pi * out["weekday"] / 7).round(5)

    out["dayofyear"] = out["Datetime"].dt.dayofyear
    out["day_of_year_sin"] = np.sin(2 * np.pi * out["dayofyear"] / 365).round(5)
    out["day_of_year_cos"] = np.cos(2 * np.pi * out["dayofyear"] / 365).round(5)

    sunrise_minutes = (
        out[sunrise_col].dt.hour * 60
        + out[sunrise_col].dt.minute
        + out[sunrise_col].dt.second / 60.0
    )

    if daylen_col is not None:
        day_length_minutes = (
            out[daylen_col].dt.hour * 60
            + out[daylen_col].dt.minute
            + out[daylen_col].dt.second / 60.0
        )
    else:
        day_length_minutes = (out[sunset_col] - out[sunrise_col]).dt.total_seconds() / 60.0

    total_minutes = (
        out["Datetime"].dt.hour * 60
        + out["Datetime"].dt.minute
        + out["Datetime"].dt.second / 60.0
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        daylight_phase = (total_minutes - sunrise_minutes) / day_length_minutes
    daylight_phase = np.clip(daylight_phase, 0.0, 1.0)
    daylight_phase = np.where(day_length_minutes <= 0, 0.0, daylight_phase)

    out["daylight_intensity"] = np.sin(np.pi * daylight_phase)
    out["daylight_intensity"] = out["daylight_intensity"].fillna(0).round(5)

    drop_cols = [sunrise_col, sunset_col]
    if daylen_col is not None:
        drop_cols.append(daylen_col)
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])

    return out

import numpy as np
import pandas as pd

def add_season_features(df: pd.DataFrame,
                        datetime_col: str | None = None,
                        hemisphere: str = "north") -> pd.DataFrame:
    """
    Adds season features using *meteorological* seasons:
      - season: categorical {'winter','spring','summer','autumn'}
      - season_idx: int {0..3} in cyclic order (north: winter=0, spring=1, summer=2, autumn=3)
      - season_sin, season_cos: cyclic encoding of 4 seasons
      - day_in_season: 1..(90/91/92/92)
      - season_len: length of current season (days)
      - season_phase_sin, season_phase_cos: progress within current season (0..1) encoded on circle

    Parameters
    ----------
    df : DataFrame with a DatetimeIndex or a datetime column
    datetime_col : if None (default), use DatetimeIndex; else use this column
    hemisphere : "north" (default) or "south" (labels rotated by 2)

    Returns
    -------
    DataFrame (copy) with added columns.
    """
    out = df.copy()

    # ---- Get a pandas Series of datetimes (either from index or a column) ----
    if datetime_col is None:
        if not isinstance(out.index, pd.DatetimeIndex):
            raise ValueError("DatetimeIndex not found. Provide datetime_col or set the index to DatetimeIndex.")
        dt_series = pd.Series(out.index, index=out.index)
    else:
        dt_series = pd.to_datetime(out[datetime_col], errors="coerce")
        if dt_series.isna().any():
            raise ValueError(f"Some values in {datetime_col} could not be parsed to datetime.")
        dt_series.index = out.index  # align

    dt = dt_series.dt
    tz = dt.tz  # preserve timezone if present

    y = dt.year
    m = dt.month
    d = dt_series.dt.normalize()  # midnight same day (keeps tz if present)

    # ---- Determine season start (meteorological): Dec 1, Mar 1, Jun 1, Sep 1 ----
    # Start months per row: {12 (DJF), 3 (MAM), 6 (JJA), 9 (SON)}
    start_month = np.select(
        [
            (m == 12),
            (m >= 3) & (m <= 5),
            (m >= 6) & (m <= 8),
            (m >= 9) & (m <= 11)
        ],
        [12, 3, 6, 9],
        default=12  # covers Jan/Feb as season started in previous Dec
    ).astype(int)

    start_year = np.where(m <= 2, y - 1, y).astype(int)

    start_idx = pd.to_datetime({
        "year": start_year,
        "month": start_month,
        "day": 1
    })
    if tz is not None:
        start_idx = start_idx.tz_localize(tz)
    start_date = pd.Series(start_idx, index=out.index)

    # Next season begins 3 months later; Dec -> Mar of next year
    next_month = ((start_month % 12) + 3).astype(int)
    next_year = (start_year + (start_month == 12).astype(int)).astype(int)

    next_idx = pd.to_datetime({
        "year": next_year,
        "month": next_month,
        "day": 1
    })
    if tz is not None:
        next_idx = next_idx.tz_localize(tz)
    next_start = pd.Series(next_idx, index=out.index)

    season_len = (next_start - start_date).dt.days.astype(int) # type: ignore
    day_in_season = ((d - start_date).dt.days + 1).astype(int) # type: ignore # 1-based
    season_progress = day_in_season / season_len  # 0..1

    # ---- Season indices & labels (north canonical order) ----
    idx_map = {12: 0, 3: 1, 6: 2, 9: 3}  # winter, spring, summer, autumn
    season_idx_north = pd.Series(start_month, index=out.index).map(idx_map).astype(int)

    if hemisphere.lower().startswith("south"):
        season_idx = (season_idx_north + 2) % 4  # rotate by two seasons
    else:
        season_idx = season_idx_north

    labels_order = np.array(["winter", "spring", "summer", "autumn"])
    season_labels = pd.Categorical(labels_order[season_idx], categories=labels_order, ordered=True)

    # ---- Encodings ----
    season_sin = np.sin(2 * np.pi * season_idx / 4.0)
    season_cos = np.cos(2 * np.pi * season_idx / 4.0)
    season_phase_sin = np.sin(2 * np.pi * season_progress)
    season_phase_cos = np.cos(2 * np.pi * season_progress)

    # ---- Assign back ----
    #out["season"] = season_labels
    out["season_idx"] = season_idx.astype(int)
    out["season_sin"] = season_sin
    out["season_cos"] = season_cos
    out["day_in_season"] = day_in_season
    out["season_len"] = season_len
    out["season_phase_sin"] = season_phase_sin
    out["season_phase_cos"] = season_phase_cos

    return out

def make_lag_features(
    df: pd.DataFrame,
    targets: Optional[Iterable[str]] = None,
    *,
    target: Optional[str] = None,
    other_exclude: Optional[Iterable[str]] = None,
    lags: Iterable[int] = (1, 24, 168),
    drop_original: bool = False,
) -> pd.DataFrame:
    """Add lagged/statistical features for one or more target columns."""

    def _ensure_list(values):
        if values is None:
            return []
        if isinstance(values, str):
            return [values]
        return list(values)

    if targets is None and target is None:
        raise ValueError("Specify targets via `targets` or `target`.")
    if target is not None:
        if targets is not None:
            raise ValueError("Use either `targets` or `target`, not both.")
        targets = target

    target_list = _ensure_list(targets)
    if not target_list:
        raise ValueError("`targets` must contain at least one value.")

    out = df.copy()
    lag_list = list(lags)
    for tgt in target_list:
        if tgt not in out.columns:
            raise KeyError(f"Column '{tgt}' not found in dataframe.")

        series = out[tgt]
        feature_map: dict[str, pd.Series] = {}
        for lag in lag_list:
            feature_map[f"{tgt}_lag_{lag}"] = series.shift(lag)

        shifted = series.shift(1)
        feature_map[f"{tgt}_rmean_24"] = shifted.rolling(24, min_periods=1).mean()
        feature_map[f"{tgt}_rstd_24"] = shifted.rolling(24, min_periods=1).std()
        feature_map[f"{tgt}_rmean_168"] = shifted.rolling(168, min_periods=1).mean()
        feature_map[f"{tgt}_rstd_168"] = shifted.rolling(168, min_periods=1).std()

        feature_block = pd.DataFrame(feature_map, index=out.index)
        out = out.join(feature_block)

        new_columns = list(feature_block.columns)
        column_order = list(out.columns)
        for name in new_columns:
            column_order.remove(name)
        insert_at = column_order.index(tgt) + 1
        column_order = column_order[:insert_at] + new_columns + column_order[insert_at:]
        out = out.loc[:, column_order]

    drop_cols = set()
    if drop_original:
        drop_cols.update(target_list)
    drop_cols.update(_ensure_list(other_exclude))

    if drop_cols:
        existing = [c for c in drop_cols if c in out.columns]
        if existing:
            out = out.drop(columns=existing)

    return out

def rolling_dates(
    data: Union[pd.Series, pd.DataFrame],
    max_months: int = 12,
    test_offset_days: int = 7,
    test_days: int = 1,
) -> List[Dict]:

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("data must be indexed by a pandas.DatetimeIndex.")

    df = data.sort_index()
    if df.empty:
        return []

    # Helpers
    def midnight_on_or_after(ts: pd.Timestamp) -> pd.Timestamp:
        # Keep tz if present
        return ts.normalize() if ts.hour == 0 else (ts.normalize() + pd.Timedelta(days=1))

    def make_test_slice(train_end: pd.Timestamp):
        raw = train_end + pd.Timedelta(days=test_offset_days)
        test_start = midnight_on_or_after(raw)
        test_stop = test_start + pd.Timedelta(days=test_days)
        mask = (df.index >= test_start) & (df.index < test_stop)
        test = df.loc[mask]
        return test

    def record(train_slice: pd.DataFrame, test_slice: pd.DataFrame, scheme: str, meta: Dict) -> Dict:
        rec = {
            "m":          meta.get("m"),
            "train_start": train_slice.index[0],
            "train_end":   train_slice.index[-1],
            "test_start":  test_slice.index[0],
            "test_end":    test_slice.index[-1],
            "scheme":      scheme,
        }
        # Optional metadata (safe to ignore downstream)
        if "extra_days" in meta:
            rec["extra_days"] = meta["extra_days"]
        if "shift_days" in meta:
            rec["shift_days"] = meta["shift_days"]

        # Emphasize Jan 1 tests
        ts = rec["test_start"]
        rec["is_new_year_test"] = (ts.month == 1 and ts.day == 1)
        return rec

    base = df.index[0].normalize()  # start-of-day for month math (preserves tz if present)
    out_anchor = []
    out_rest: List[Dict] = []
    seen = set()  # de-duplicate by (train_start, test_start)

    # -----------------------------
    # Split 0 (keep exactly as-is)
    # -----------------------------
    # Train until 7 days before end of year (inclusive 23:00), test on Jan 1 next year
    train_end_0 = (pd.Period(base, freq="Y").end_time.normalize()
                   - pd.DateOffset(days=7)
                   + pd.Timedelta(hours=23))
    train_0 = df.loc[:train_end_0]
    test_0 = make_test_slice(train_end_0)
    if not test_0.empty:
        rec0 = record(train_0, test_0, scheme="anchor_new_year", meta={"m": 0})  # type: ignore
        out_anchor.append(rec0)
        seen.add((rec0["train_start"], rec0["test_start"]))

    # ---------------------------------------------------------
    # Original monthly-cumulative scheme (m = 1..max_months)
    # ---------------------------------------------------------
    for m in range(1, max_months + 1):
        train_end = (base + pd.DateOffset(months=m)) - pd.Timedelta(hours=1)
        if train_end > df.index[-1]:
            break  # not enough data even for training
        train = df.loc[:train_end]

        test = make_test_slice(train_end)
        if test.empty:
            continue  # not enough future data for this split

        key = (train.index[0], test.index[0])
        if key in seen:
            continue
        rec = record(train, test, scheme="cumulative_months", meta={"m": m})  # type: ignore
        out_rest.append(rec)
        seen.add(key)

    # -------------------------------------------------------------------
    # A) Fixed 1-month training window, step forward by 7 days
    # -------------------------------------------------------------------
    step = pd.Timedelta(days=7)
    # First one-month window ends one hour before (base + 1 month)
    end_one_month = (base + pd.DateOffset(months=1)) - pd.Timedelta(hours=1)
    train_end = end_one_month

    while True:
        # Training is the last 1 calendar month ending at train_end (inclusive)
        # Define train_start as (train_end + 1h) - 1 month to get an exact 1-month window
        train_start = (train_end + pd.Timedelta(hours=1)) - pd.DateOffset(months=1)

        # If the window starts before we have data, we can still try using what's available,
        # but require at least some coverage.
        train = df.loc[train_start:train_end]
        if train.empty:
            # If even the first window is empty, or we've slid past available data, stop.
            break

        test = make_test_slice(train_end)
        if test.empty:
            # Once test becomes empty, further steps will also be empty -> stop
            break

        key = (train.index[0], test.index[0])
        if key not in seen:
            rec = record(train, test, scheme="fixed_1m_step7d",  # type: ignore
                         meta={"m": train_end.month-1, "shift_days": (train_end - end_one_month).days})
            out_rest.append(rec)
            seen.add(key)

        # Shift forward by 1 week
        train_end = train_end + step
        # Guard: don't let training end go past the very last timestamp by too much
        if train_end > (df.index[-1] + pd.Timedelta(days=test_offset_days + test_days + 1)):
            break

    # -------------------------------------------------------------------
    # B) Cumulative months + {+1..+6} extra days
    # -------------------------------------------------------------------
    for m in range(1, max_months + 1):
        base_m_end = (base + pd.DateOffset(months=m)) - pd.Timedelta(hours=1)
        for d in range(1, 7):  # +1..+6 extra days
            train_end = base_m_end + pd.Timedelta(days=d)
            if train_end > df.index[-1]:
                break
            train = df.loc[:train_end]
            if train.empty:
                continue
            test = make_test_slice(train_end)
            if test.empty:
                continue
            key = (train.index[0], test.index[0])
            if key in seen:
                continue
            rec = record(train, test, scheme="cum_m_plus_days",  # type: ignore
                         meta={"m": m, "extra_days": d})
            out_rest.append(rec)
            seen.add(key)

    # Final order: keep anchor first, then others sorted by test_start
    out_rest_sorted = sorted(out_rest, key=lambda r: r["test_start"])
    return out_anchor + out_rest_sorted
 
def nan_outliers(s: pd.Series):
    decomp_ = MSTL(s, periods=(24,24*7),lmbda=None)
    res_dec_ = decomp_.fit()
    s = s.astype(float).copy()
    mask = (abs(res_dec_.resid) > 550) | (s < 100) | ((operator.sub(res_dec_.trend, res_dec_.observed)/res_dec_.observed) > 1)

    s.loc[mask] = np.nan
    out = s
    return out

def fill_nan_with_sarima(df: pd.DataFrame,
                         time_col: str,
                         value_col: str,
                         order=(3, 1, 1),
                         seasonal_order=(2, 1, 1, 24),
                         trend='ct') -> pd.DataFrame:

    if time_col not in df.columns:
        raise KeyError(f"Column '{time_col}' not found in dataframe.")
    if value_col not in df.columns:
        raise KeyError(f"Column '{value_col}' not found in dataframe.")

    working = df.copy()
    working[time_col] = pd.to_datetime(working[time_col], errors="coerce")
    if working[time_col].isna().any():
        raise ValueError(f"Failed to parse datetime values in column '{time_col}'.")

    working = (
        working
        .sort_values(time_col)
        .drop_duplicates(subset=time_col, keep='last')
        .set_index(time_col)
    )

    working = working.asfreq('h')

    y = pd.to_numeric(working[value_col], errors='coerce')
    if y.notna().sum() == 0:
        raise ValueError(f"Column '{value_col}' does not contain any finite values for SARIMA fitting.")

    mod = SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    y_filled = y.copy()
    na_mask = (y_filled.isna()) | (y_filled.isnull()) | (y_filled == 0)
        
    res = mod.fit(disp=False)
    #print("fitted")
    pred_mean = res.predict(start=y.index[0], end=y.index[-1], dynamic=False)# type: ignore
    
    if na_mask.any():
        y_filled.loc[na_mask] = pred_mean.loc[na_mask]

    filled_col = f"{value_col}_filled"
    predict_col = f"{value_col}_predicted"
    working[value_col] = y
    working[filled_col] = y_filled
    working[predict_col] = pred_mean

    cols = list(working.columns) # type: ignore
    cols.remove(filled_col)
    if value_col in cols:
        insert_at = cols.index(value_col) + 1
    else:
        insert_at = len(cols)
    cols = cols[:insert_at] + [filled_col] + [predict_col] + cols[insert_at:]
    working = working.loc[:, cols]
    
    return working.reset_index()


if __name__ == "__main__":
    read_P21 = pd.read_excel(r"data\PD_2021History_GMT2.xlsx",sheet_name='Sheet1',usecols=[0, 1])
    read_P22 = pd.read_excel(r"data\PD_2022History_GMT2.xlsx",sheet_name='Sheet1',usecols=[0, 1])
    read_Q21 = pd.read_excel(r"data\QD_2021History_GMT2.xlsx",sheet_name='Sheet1',usecols=[0, 1])
    read_Q22 = pd.read_excel(r"data\QD_2022History_GMT2.xlsx",sheet_name='Sheet1',usecols=[0, 1])
    read_Weather = pd.read_excel(r"data\PD_Weather_GMT2.xlsx",sheet_name='Sheet1')
    read_Daylight = pd.read_excel(r"data\Sunrise_sunset.xlsx",sheet_name='Sheet1')    

    merged_PQ21 = pd.concat([read_P21, read_Q21], axis=1, join="outer")
    merged_PQ22 = pd.concat([read_P22, read_Q22], axis=1, join="outer")
    merged_PQ = pd.concat([merged_PQ21, merged_PQ22])
    merged_PQ = merged_PQ.loc[:,~merged_PQ.columns.duplicated()].copy()

    merged_PQ = pd.merge(merged_PQ, read_Weather, on="Datetime", how="inner")
    merged_PQ = pd.merge(merged_PQ, solar_features(read_Daylight), on="Datetime", how="inner")
    merged_PQ = add_season_features(merged_PQ, datetime_col="Datetime", hemisphere="north")
    
    PQ = localize_fixed_utc2(merged_PQ, explicit_format="%Y-%m-%d %H:%M:%S")
    PQ = pd.concat([PQ.set_index('Datetime'), build_holiday_features(PQ.set_index('Datetime').index)], axis=1, join="outer")    

    res_dec_P = nan_outliers(PQ["P_Power"])
    res_dec_Q = nan_outliers(PQ["Q_Power"])
    PQ["P_Power"] = res_dec_P
    PQ["Q_Power"] = res_dec_Q    

    P =  fill_nan_with_sarima(PQ.reset_index(names='Datetime'),time_col='Datetime', value_col='P_Power')
    Q =  fill_nan_with_sarima(PQ.reset_index(names='Datetime'),time_col='Datetime', value_col='Q_Power')
    PQ["P_Power"] = pd.Series(P["P_Power_filled"].to_numpy(), index=PQ.index)
    PQ["Q_Power"] = pd.Series(Q["Q_Power_filled"].to_numpy(), index=PQ.index)
    
    PQ.index = PQ.index.tz_localize(None) # type: ignore
    DataEnd = pd.Period(PQ.index[-1].normalize(), freq='Y').start_time.normalize()
    mask = PQ.index < DataEnd
    lags=[1, 24, 168]
    InputLag:List = []
    for i, lag in enumerate(lags):
        InputLag.append(lag)
        Data = PQ.copy()        
        Data = make_lag_features(Data, targets=["P_Power", "Q_Power"], lags=InputLag, drop_original=False)
        with pd.ExcelWriter(rf"data\PQ.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            Data.dropna().to_excel(writer, sheet_name=f"{lag}")        
        splits = rolling_dates(Data.loc[mask].dropna()) 
        with pd.ExcelWriter(rf"data\splits.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            pd.DataFrame(splits).drop(columns="m").to_excel(writer, sheet_name=f"{lag}")


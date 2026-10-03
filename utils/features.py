from config.config import settings
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
import numpy as np

ANCHOR_YEARS = settings.ANCHOR_YEARS
TEMP_COLS = settings.TEMP_COLS
GHI_COLS = settings.GHI_COLS

WEATHER = settings.WEATHER

T_BASE = settings.T_BASE

def _attach_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Anchor (Year, Month, Day, Hour) to real calendar timestamps.

    The competition files use Hour = 1..24 (hour-*ending* labels); the
    timestamp is built at the hour *beginning* (Hour-1) so that all 24
    hourly rows of a calendar day fall on that day - which keeps
    day-of-week, weekend and holiday flags exactly aligned.
    """
    df = df.copy()
    cal_year = df["Year"].map(ANCHOR_YEARS)
    hour = df["Hour"]
    if hour.max() == 24:  # hour-ending 1..24 -> hour-beginning 0..23
        hour = hour - 1
    df["Date"] = pd.to_datetime(
        dict(year=cal_year, month=df["Month"], day=df["Day"], hour=hour)
    )
    return df

def build_features(df_train, df_test, pls_fit_years = [1, 2]):

    frames = [df_train] if df_test is None else [df_train, df_test]
    full_df = pd.concat(frames, ignore_index=True)
    full_df = _attach_datetime(full_df).sort_values("Date").reset_index(drop=True)

    fit_mask = full_df["Year"].isin(pls_fit_years) & full_df["Load"].notna()
    X_temp_fit = full_df.loc[fit_mask, TEMP_COLS]
    X_ghi_fit = full_df.loc[fit_mask, GHI_COLS]
    y_fit = full_df.loc[fit_mask, "Load"]

    pls_temp = PLSRegression(n_components=1)
    pls_temp.fit(X_temp_fit, y_fit)
    full_df["Combined_Temp"] = pls_temp.transform(full_df[TEMP_COLS])[:, 0]

    pls_ghi = PLSRegression(n_components=2)
    pls_ghi.fit(X_ghi_fit, y_fit)
    full_df["Combined_GHI_1"] = pls_ghi.transform(full_df[GHI_COLS])[:, 0]
    full_df["Combined_GHI_2"] = pls_ghi.transform(full_df[GHI_COLS])[:, 1]

    t0 = full_df["Date"].iloc[0]
    t = (full_df["Date"] - t0).dt.total_seconds() / 3600.0

    full_df["sin_day_k1"] = np.sin(2 * np.pi * 1 * t / 24)
    full_df["cos_day_k1"] = np.cos(2 * np.pi * 1 * t / 24)
    full_df["sin_day_k2"] = np.sin(2 * np.pi * 2 * t / 24)
    full_df["cos_day_k2"] = np.cos(2 * np.pi * 2 * t / 24)
    full_df["sin_week_k1"] = np.sin(2 * np.pi * 1 * t / 168.0)
    full_df["cos_week_k1"] = np.cos(2 * np.pi * 1 * t / 168.0)
    full_df["sin_year_k3"] = np.sin(2 * np.pi * 3 * t / 8766.0)
    full_df["cos_year_k3"] = np.cos(2 * np.pi * 3 * t / 8766.0)
    full_df["dow"] = full_df["Date"].dt.dayofweek
    full_df["is_weekend"] = (full_df["dow"] >= 5).astype(int)

    for stem, col in (("temp", "Combined_Temp"), ("ghi_1", "Combined_GHI_1"), ("ghi_2", "Combined_GHI_2")):
        full_df[f"lag_1_{stem}"] = full_df[col].shift(1)
        full_df[f"delta_1_{stem}"] = full_df[col].diff(1)
        full_df[f"delta_24_{stem}"] = full_df[col].diff(24)

    mean_temp_c = full_df[TEMP_COLS].mean(axis=1)
    full_df["CDH"] = (mean_temp_c - T_BASE).clip(lower=0.0)
    full_df["HDH"] = (T_BASE - mean_temp_c).clip(lower=0.0)

    train_df = full_df.loc[fit_mask]
    test_df = full_df.loc[~fit_mask]

    return train_df, test_df
    



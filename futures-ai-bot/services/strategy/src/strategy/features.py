"""Feature engineering pipeline for short-horizon futures signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class FeatureConfig:
    """Window sizes and feature options."""

    vol_window: int = 20
    atr_window: int = 14
    breakout_window: int = 20
    zscore_window: int = 30
    trend_window_fast: int = 10
    trend_window_slow: int = 30
    opening_range_bars: int = 30


def build_feature_frame(bars: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Compute model features without forward-looking leakage."""

    cfg = config or FeatureConfig()
    if bars.empty:
        return bars.copy()

    required = {"timestamp", "symbol_root", "open", "high", "low", "close", "volume"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"Missing required bar columns for features: {missing}")

    frame = bars.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values(["symbol_root", "timestamp"]).reset_index(drop=True)

    out = []
    for _, grp in frame.groupby("symbol_root", sort=False):
        out.append(_single_symbol_features(grp.copy(), cfg))
    features = pd.concat(out, ignore_index=True)
    features = _add_cross_market_features(features)
    return features.sort_values(["symbol_root", "timestamp"]).reset_index(drop=True)


def _single_symbol_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    epsilon = 1e-9
    prev_close = df["close"].shift(1)
    range_ = df["high"] - df["low"]
    body = (df["close"] - df["open"]).abs()

    df["ret_1"] = df["close"].pct_change()
    df["log_ret_1"] = np.log(df["close"] / prev_close.replace(0, np.nan))
    df["rolling_vol"] = df["ret_1"].rolling(cfg.vol_window, min_periods=5).std()
    df["true_range"] = pd.concat(
        [(df["high"] - df["low"]), (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    df["atr"] = df["true_range"].rolling(cfg.atr_window, min_periods=5).mean()

    typical = (df["high"] + df["low"] + df["close"]) / 3
    date_key = df["timestamp"].dt.tz_convert("America/New_York").dt.date
    vwap_num = (typical * df["volume"]).groupby(date_key).cumsum()
    vwap_den = df["volume"].groupby(date_key).cumsum().replace(0, np.nan)
    df["vwap"] = vwap_num / vwap_den
    df["dist_from_vwap"] = (df["close"] - df["vwap"]) / (df["vwap"] + epsilon)

    df["candle_body_ratio"] = body / (range_ + epsilon)
    df["candle_range_ratio"] = range_ / (
        df["close"].rolling(cfg.vol_window, min_periods=5).mean() + epsilon
    )
    df["close_location"] = (df["close"] - df["low"]) / (range_ + epsilon)
    df["upper_wick_ratio"] = (df["high"] - df[["open", "close"]].max(axis=1)) / (range_ + epsilon)
    df["lower_wick_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / (range_ + epsilon)

    prior_high = df["high"].shift(1).rolling(cfg.breakout_window, min_periods=5).max()
    prior_low = df["low"].shift(1).rolling(cfg.breakout_window, min_periods=5).min()
    df["breakout_pressure"] = (df["close"] - prior_high) / (df["atr"] + epsilon)
    df["breakdown_pressure"] = (prior_low - df["close"]) / (df["atr"] + epsilon)

    rolling_range_mean = range_.rolling(cfg.breakout_window, min_periods=5).mean()
    df["compression_expansion"] = range_ / (rolling_range_mean + epsilon)

    roll_mean = df["close"].rolling(cfg.zscore_window, min_periods=10).mean()
    roll_std = df["close"].rolling(cfg.zscore_window, min_periods=10).std()
    df["mean_reversion_z"] = (df["close"] - roll_mean) / (roll_std + epsilon)

    fast = df["close"].ewm(span=cfg.trend_window_fast, adjust=False).mean()
    slow = df["close"].ewm(span=cfg.trend_window_slow, adjust=False).mean()
    df["trend_slope_proxy"] = (fast - slow) / (df["atr"] + epsilon)
    df["trend_strength"] = df["close"].diff(cfg.trend_window_fast) / (df["atr"] + epsilon)

    local_date = df["timestamp"].dt.tz_convert("America/New_York").dt.date
    opening_range_high = df.groupby(local_date)["high"].transform(
        lambda s: s.iloc[: cfg.opening_range_bars].max() if len(s) else np.nan
    )
    opening_range_low = df.groupby(local_date)["low"].transform(
        lambda s: s.iloc[: cfg.opening_range_bars].min() if len(s) else np.nan
    )
    df["opening_range_position"] = (df["close"] - opening_range_low) / (
        (opening_range_high - opening_range_low) + epsilon
    )

    ny = df["timestamp"].dt.tz_convert("America/New_York")
    df["hour"] = ny.dt.hour
    df["minute"] = ny.dt.minute
    df["minute_of_day"] = (df["hour"] * 60) + df["minute"]
    df["day_of_week"] = ny.dt.dayofweek
    df["is_us_open_session"] = ((df["hour"] > 9) | ((df["hour"] == 9) & (df["minute"] >= 30))) & (
        df["hour"] < 16
    )

    df["range_pct"] = range_ / (df["close"].shift(1).abs() + epsilon)
    df["volume_change"] = df["volume"].pct_change()
    df["volume_z"] = (df["volume"] - df["volume"].rolling(cfg.vol_window, min_periods=5).mean()) / (
        df["volume"].rolling(cfg.vol_window, min_periods=5).std() + epsilon
    )
    return df


def _add_cross_market_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rel_es_vs_nq"] = np.nan
    out["rel_cl_vs_gc"] = np.nan
    out["spread_ret_es_nq"] = np.nan
    out["spread_ret_cl_gc"] = np.nan

    out = _join_pair(
        out=out,
        left_symbol="ES",
        right_symbol="NQ",
        rel_col="rel_es_vs_nq",
        spread_col="spread_ret_es_nq",
    )
    out = _join_pair(
        out=out,
        left_symbol="CL",
        right_symbol="GC",
        rel_col="rel_cl_vs_gc",
        spread_col="spread_ret_cl_gc",
    )
    return out


def _join_pair(
    out: pd.DataFrame,
    left_symbol: str,
    right_symbol: str,
    rel_col: str,
    spread_col: str,
) -> pd.DataFrame:
    right = out[out["symbol_root"] == right_symbol][["timestamp", "close", "ret_1"]].rename(
        columns={"close": "pair_close", "ret_1": "pair_ret"}
    )
    mask = out["symbol_root"] == left_symbol
    left = out.loc[mask, ["timestamp", "close", "ret_1"]].copy()
    left["_idx"] = left.index
    merged = left.merge(right, on="timestamp", how="left")
    merged[rel_col] = merged["close"] / merged["pair_close"]
    merged[spread_col] = merged["ret_1"] - merged["pair_ret"]
    out.loc[merged["_idx"], rel_col] = merged[rel_col].values
    out.loc[merged["_idx"], spread_col] = merged[spread_col].values
    return out

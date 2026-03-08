"""Label generation utilities with strict future-window boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class LabelConfig:
    """Parameters for horizon and stop/target construction."""

    forecast_horizon: int = 5
    barrier_horizon: int = 12
    target_atr_mult: float = 1.0
    stop_atr_mult: float = 1.0


def add_labels(frame: pd.DataFrame, config: LabelConfig | None = None) -> pd.DataFrame:
    """Append direction/MFE/MAE/meta labels per symbol."""

    cfg = config or LabelConfig()
    required = {"timestamp", "symbol_root", "close", "high", "low", "atr"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for labeling: {missing}")

    df = frame.sort_values(["symbol_root", "timestamp"]).copy()
    labeled = []
    for _, grp in df.groupby("symbol_root", sort=False):
        labeled.append(_label_symbol(grp.copy(), cfg))
    return pd.concat(labeled, ignore_index=True)


def _label_symbol(df: pd.DataFrame, cfg: LabelConfig) -> pd.DataFrame:
    n = len(df)
    closes = df["close"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    atr = df["atr"].ffill().bfill().fillna(0.0).to_numpy(dtype=float)

    next_direction = np.full(n, np.nan)
    target_before_stop = np.full(n, np.nan)
    mfe = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    meta = np.full(n, np.nan)

    for i in range(n):
        forecast_idx = i + cfg.forecast_horizon
        if forecast_idx >= n:
            continue
        future_close = closes[forecast_idx]
        direction = 1.0 if future_close > closes[i] else 0.0
        next_direction[i] = direction

        lookahead_end = min(i + 1 + cfg.barrier_horizon, n)
        if i + 1 >= lookahead_end:
            continue
        win_high = highs[i + 1 : lookahead_end]
        win_low = lows[i + 1 : lookahead_end]
        if len(win_high) == 0:
            continue

        direction_sign = 1 if direction == 1.0 else -1
        barrier = max(atr[i], 1e-9)
        target = closes[i] + (direction_sign * cfg.target_atr_mult * barrier)
        stop = closes[i] - (direction_sign * cfg.stop_atr_mult * barrier)

        target_idx = None
        stop_idx = None
        for j in range(len(win_high)):
            if direction_sign > 0:
                if win_high[j] >= target and target_idx is None:
                    target_idx = j
                if win_low[j] <= stop and stop_idx is None:
                    stop_idx = j
            else:
                if win_low[j] <= target and target_idx is None:
                    target_idx = j
                if win_high[j] >= stop and stop_idx is None:
                    stop_idx = j
            if target_idx is not None and stop_idx is not None:
                break

        if target_idx is None and stop_idx is None:
            target_before_stop[i] = 0.0
        elif target_idx is None:
            target_before_stop[i] = 0.0
        elif stop_idx is None:
            target_before_stop[i] = 1.0
        else:
            target_before_stop[i] = 1.0 if target_idx < stop_idx else 0.0

        if direction_sign > 0:
            mfe_raw = np.max(win_high - closes[i])
            mae_raw = np.min(win_low - closes[i])
        else:
            mfe_raw = np.max(closes[i] - win_low)
            mae_raw = np.min(closes[i] - win_high)
        mfe[i] = mfe_raw
        mae[i] = mae_raw
        meta[i] = 1.0 if (target_before_stop[i] == 1.0 and mfe_raw > abs(mae_raw)) else 0.0

    df["next_n_bar_direction"] = next_direction
    df["target_hit_before_stop"] = target_before_stop
    df["max_favorable_excursion"] = mfe
    df["max_adverse_excursion"] = mae
    df["meta_label_trade_quality"] = meta
    return df

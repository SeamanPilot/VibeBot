"""Synthetic futures data generation for local demos and tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from common.data import save_normalized_bars
from common.metadata import load_futures_metadata
from common.settings import get_settings


def _round_to_tick(value: float, tick_size: float) -> float:
    return round(round(value / tick_size) * tick_size, 10)


def generate_symbol_bars(
    symbol_root: str,
    start: str,
    periods: int,
    freq: str,
    seed: int,
    base_price: float,
    tick_size: float,
) -> pd.DataFrame:
    """Generate synthetic OHLCV bars from a drift+noise process."""

    rng = np.random.default_rng(seed)
    index = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")
    close = np.empty(periods)
    close[0] = base_price
    drift = rng.normal(loc=0.0, scale=tick_size * 0.5, size=periods)
    shock = rng.normal(loc=0.0, scale=tick_size * 2.0, size=periods)
    for i in range(1, periods):
        close[i] = max(tick_size, close[i - 1] + drift[i] + shock[i])

    open_ = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(open_, close) + np.abs(rng.normal(scale=tick_size * 2.2, size=periods))
    low = np.minimum(open_, close) - np.abs(rng.normal(scale=tick_size * 2.2, size=periods))
    volume = rng.integers(low=50, high=2000, size=periods)

    frame = pd.DataFrame(
        {
            "timestamp": index,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume.astype(float),
        }
    )
    for col in ["open", "high", "low", "close"]:
        frame[col] = frame[col].map(lambda v: _round_to_tick(float(v), tick_size))
    frame["high"] = frame[["open", "close", "high"]].max(axis=1)
    frame["low"] = frame[["open", "close", "low"]].min(axis=1)
    return frame


def _resample_to_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    bars = df_1m.copy()
    bars = bars.set_index("timestamp")
    agg = bars.resample("5min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return agg.dropna().reset_index()


def create_sample_dataset(output_dir: Path, periods_1m: int = 1500, seed: int = 7) -> None:
    """Create synthetic sample bars for all configured roots and 1m/5m timeframes."""

    settings = get_settings()
    metadata = load_futures_metadata(settings.futures_metadata_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_price_map = {
        "ES": 5100.0,
        "NQ": 18200.0,
        "CL": 74.0,
        "GC": 2130.0,
        "MES": 5100.0,
        "MNQ": 18200.0,
        "MCL": 74.0,
        "MGC": 2130.0,
    }

    for idx, symbol in enumerate(metadata.symbols):
        root = symbol.root
        bars_1m = generate_symbol_bars(
            symbol_root=root,
            start="2025-01-02 00:00:00+00:00",
            periods=periods_1m,
            freq="1min",
            seed=seed + idx,
            base_price=base_price_map[root],
            tick_size=symbol.tick_size,
        )
        bars_1m["symbol_root"] = root
        bars_1m["timeframe"] = "1m"
        bars_1m["source"] = "synthetic"
        save_normalized_bars(
            bars_1m[
                [
                    "timestamp",
                    "symbol_root",
                    "timeframe",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "source",
                ]
            ],
            output_dir / f"{root}_1m.parquet",
        )

        bars_5m = _resample_to_5m(bars_1m[["timestamp", "open", "high", "low", "close", "volume"]])
        bars_5m["symbol_root"] = root
        bars_5m["timeframe"] = "5m"
        bars_5m["source"] = "synthetic"
        save_normalized_bars(
            bars_5m[
                [
                    "timestamp",
                    "symbol_root",
                    "timeframe",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "source",
                ]
            ],
            output_dir / f"{root}_5m.parquet",
        )


def main() -> None:
    """CLI for generating sample data files."""

    parser = argparse.ArgumentParser(description="Generate synthetic futures bars.")
    parser.add_argument("--output-dir", default="data/sample", type=str)
    parser.add_argument("--periods-1m", default=1500, type=int)
    parser.add_argument("--seed", default=7, type=int)
    args = parser.parse_args()
    create_sample_dataset(Path(args.output_dir), periods_1m=args.periods_1m, seed=args.seed)


if __name__ == "__main__":
    main()

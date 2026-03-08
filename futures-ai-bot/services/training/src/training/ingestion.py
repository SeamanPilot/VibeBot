"""OHLCV ingestion entrypoints for research datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from common.data import load_bar_file, normalize_bars, save_normalized_bars


def ingest_file(
    input_path: Path,
    output_path: Path,
    symbol_root: str,
    timeframe: str,
    source: str = "csv",
) -> pd.DataFrame:
    """Read, validate, normalize, and persist an OHLCV file."""

    bars = load_bar_file(
        path=input_path, symbol_root=symbol_root, timeframe=timeframe, source=source
    )
    save_normalized_bars(bars, output_path)
    return bars


def ingest_dataframe(
    frame: pd.DataFrame,
    output_path: Path,
    symbol_root: str,
    timeframe: str,
    source: str = "dataframe",
) -> pd.DataFrame:
    """Normalize bars from an in-memory DataFrame."""

    bars = normalize_bars(frame, symbol_root=symbol_root, timeframe=timeframe, source=source)
    save_normalized_bars(bars, output_path)
    return bars

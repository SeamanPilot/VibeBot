"""Data ingestion and normalization helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_BAR_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}


def load_bar_file(path: Path, symbol_root: str, timeframe: str, source: str) -> pd.DataFrame:
    """Load bars from CSV or parquet and normalize schema."""

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    return normalize_bars(df=df, symbol_root=symbol_root, timeframe=timeframe, source=source)


def normalize_bars(df: pd.DataFrame, symbol_root: str, timeframe: str, source: str) -> pd.DataFrame:
    """Enforce canonical bar schema used by strategy/training/execution services."""

    missing = REQUIRED_BAR_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required bar columns: {missing}")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["symbol_root"] = symbol_root
    out["timeframe"] = timeframe
    out["source"] = source
    out = out[
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
    ]
    out = out.sort_values("timestamp").drop_duplicates(["timestamp", "symbol_root", "timeframe"])
    numeric_cols = ["open", "high", "low", "close", "volume"]
    out[numeric_cols] = out[numeric_cols].astype(float)
    return out.reset_index(drop=True)


def load_bar_directory(
    directory: Path,
    symbol_root: str,
    timeframe: str,
    source: str,
) -> pd.DataFrame:
    """Load and concatenate all bar files in a directory."""

    files = sorted(
        [*directory.glob("*.csv"), *directory.glob("*.parquet"), *directory.glob("*.pq")]
    )
    if not files:
        raise FileNotFoundError(f"No bar files found in {directory}")
    frames = [
        load_bar_file(path=file, symbol_root=symbol_root, timeframe=timeframe, source=source)
        for file in files
    ]
    return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def save_normalized_bars(df: pd.DataFrame, output_path: Path) -> None:
    """Save normalized bars to parquet or CSV based on suffix."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        df.to_csv(output_path, index=False)
        return
    if output_path.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(output_path, index=False)
        return
    raise ValueError(f"Unsupported output format: {output_path}")

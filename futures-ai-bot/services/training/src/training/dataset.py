"""Dataset builder with leakage-safe walk-forward splits."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from training.labels import LabelConfig, add_labels


@dataclass(slots=True)
class SplitConfig:
    """Walk-forward split settings measured in unique bars."""

    train_size: int = 500
    validation_size: int = 150
    test_size: int = 150
    step_size: int = 150


@dataclass(slots=True)
class DatasetBuildConfig:
    """Dataset build configuration."""

    label: LabelConfig = field(default_factory=LabelConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    random_seed: int = 7


class DatasetBuilder:
    """Build feature+label datasets and deterministic walk-forward splits."""

    def __init__(self, config: DatasetBuildConfig | None = None) -> None:
        self.config = config or DatasetBuildConfig()

    def build(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        """Create a labeled, cleaned dataset with deterministic ordering."""

        labeled = add_labels(feature_frame, self.config.label)
        labeled = labeled.sort_values(["symbol_root", "timestamp"]).reset_index(drop=True)
        cleaned = labeled.dropna(subset=["next_n_bar_direction", "target_hit_before_stop"])
        return cleaned.reset_index(drop=True)

    def build_splits(self, dataset: pd.DataFrame) -> list[dict[str, np.ndarray]]:
        """Generate walk-forward splits based on time index to prevent leakage."""

        if dataset.empty:
            return []
        ordered = dataset.sort_values("timestamp").reset_index(drop=True)
        unique_times = ordered["timestamp"].drop_duplicates().to_numpy()

        cfg = self.config.split
        required = cfg.train_size + cfg.validation_size + cfg.test_size
        splits: list[dict[str, np.ndarray]] = []
        start = 0
        while start + required <= len(unique_times):
            train_end = start + cfg.train_size
            val_end = train_end + cfg.validation_size
            test_end = val_end + cfg.test_size

            train_times = set(unique_times[start:train_end])
            val_times = set(unique_times[train_end:val_end])
            test_times = set(unique_times[val_end:test_end])
            train_idx = np.where(ordered["timestamp"].isin(train_times))[0]
            val_idx = np.where(ordered["timestamp"].isin(val_times))[0]
            test_idx = np.where(ordered["timestamp"].isin(test_times))[0]
            splits.append({"train": train_idx, "validation": val_idx, "test": test_idx})
            start += cfg.step_size
        return splits

    def split_last(self, dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return the latest walk-forward split for training usage."""

        ordered = dataset.sort_values("timestamp").reset_index(drop=True)
        splits = self.build_splits(ordered)
        if not splits:
            raise ValueError("Not enough data for configured split sizes")
        split = splits[-1]
        train = ordered.iloc[split["train"]].reset_index(drop=True)
        validation = ordered.iloc[split["validation"]].reset_index(drop=True)
        test = ordered.iloc[split["test"]].reset_index(drop=True)
        return train, validation, test

    def save(self, dataset: pd.DataFrame, output_path: Path) -> None:
        """Persist dataset as parquet for reproducible downstream use."""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(output_path, index=False)

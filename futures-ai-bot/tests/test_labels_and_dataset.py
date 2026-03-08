from pathlib import Path

import pandas as pd
from common.data import load_bar_file
from strategy.features import build_feature_frame
from training.dataset import DatasetBuilder


def test_dataset_builder_produces_labels() -> None:
    es = load_bar_file(
        Path("data/sample/ES_1m.csv"), symbol_root="ES", timeframe="1m", source="sample"
    )
    nq = load_bar_file(
        Path("data/sample/NQ_1m.csv"), symbol_root="NQ", timeframe="1m", source="sample"
    )
    bars = pd.concat([es, nq], ignore_index=True)
    features = build_feature_frame(bars)
    builder = DatasetBuilder()
    dataset = builder.build(features)
    assert "next_n_bar_direction" in dataset.columns
    assert "target_hit_before_stop" in dataset.columns
    assert len(dataset) > 0

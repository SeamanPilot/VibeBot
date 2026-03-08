from pathlib import Path

import pandas as pd
from common.data import load_bar_file
from strategy.features import build_feature_frame


def test_csv_ingestion_normalizes_schema() -> None:
    frame = load_bar_file(
        Path("data/sample/ES_1m.csv"),
        symbol_root="ES",
        timeframe="1m",
        source="sample_csv",
    )
    assert list(frame.columns) == [
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
    assert frame["symbol_root"].iloc[0] == "ES"


def test_feature_pipeline_includes_cross_market_features() -> None:
    es = load_bar_file(
        Path("data/sample/ES_1m.csv"), symbol_root="ES", timeframe="1m", source="sample"
    )
    nq = load_bar_file(
        Path("data/sample/NQ_1m.csv"), symbol_root="NQ", timeframe="1m", source="sample"
    )
    bars = pd.concat([es, nq], ignore_index=True)
    features = build_feature_frame(bars)
    assert "rolling_vol" in features.columns
    assert "mean_reversion_z" in features.columns
    es_rows = features[features["symbol_root"] == "ES"]
    assert es_rows["rel_es_vs_nq"].notna().any()

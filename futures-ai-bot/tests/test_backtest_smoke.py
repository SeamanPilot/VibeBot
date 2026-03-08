from pathlib import Path

import pandas as pd
from backtest.engine import EventDrivenBacktester
from common.data import load_bar_file
from common.metadata import load_futures_metadata, metadata_by_root
from executor.paper_broker import PaperBroker
from executor.risk import RiskEngine
from strategy.features import build_feature_frame
from strategy.signal_engine import SignalEngine


def test_backtest_smoke_runs() -> None:
    metadata = metadata_by_root(load_futures_metadata(Path("config/futures_metadata.yaml")))
    es = load_bar_file(
        Path("data/sample/ES_1m.csv"), symbol_root="ES", timeframe="1m", source="sample"
    )
    features = build_feature_frame(es)
    data = features.copy()
    data["probability_long"] = 0.6
    data["regime"] = 1
    data["ensemble_score"] = 0.2
    data["confidence"] = 0.7
    broker = PaperBroker(metadata)
    risk = RiskEngine(metadata)
    engine = EventDrivenBacktester(
        metadata, broker=broker, signal_engine=SignalEngine(metadata), risk_engine=risk
    )
    result = engine.run(data.fillna(0.0))
    assert isinstance(result.equity_curve, pd.DataFrame)
    assert result.metrics.exposure_time >= 0.0

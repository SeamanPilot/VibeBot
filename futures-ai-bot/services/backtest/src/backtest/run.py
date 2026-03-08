"""Backtest CLI entrypoint."""

from __future__ import annotations

import argparse
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
from common.metadata import load_futures_metadata, metadata_by_root
from common.settings import get_settings
from executor.paper_broker import PaperBroker
from executor.risk import RiskConfig, RiskEngine
from strategy.features import build_feature_frame
from strategy.signal_engine import SignalEngine
from training.modeling import ensemble_signal_score
from training.registry import get_active_records, list_records, load_record
from training.synthetic_data import create_sample_dataset
from training.train import load_sample_bars

from backtest.engine import EventDrivenBacktester


def main() -> None:
    parser = argparse.ArgumentParser(description="Run event-driven backtest")
    parser.add_argument("--reports-dir", default="reports", type=str)
    parser.add_argument("--timeframe", default="1m", type=str)
    args = parser.parse_args()

    settings = get_settings()
    sample_dir = settings.data_dir / "sample"
    if not sample_dir.exists() or not list(sample_dir.glob("*_1m.*")):
        create_sample_dataset(sample_dir)

    active = get_active_records(settings.model_registry_dir)
    selected = active["production"] or active["shadow"]
    if selected is None:
        records = list_records(settings.model_registry_dir)
        if not records:
            raise RuntimeError("No models in registry. Run training first.")
        selected = records[0]

    record = load_record(settings.model_registry_dir, selected.model_id, selected.version)
    model = _load_pickle(Path(record.artifact_path))
    regime_model = _load_pickle(Path(record.regime_artifact_path))

    bars = load_sample_bars(sample_dir, timeframe=args.timeframe)
    features = build_feature_frame(bars)
    data = features.copy()
    data["probability_long"] = model.predict_proba(
        data[record.feature_columns].fillna(0.0).to_numpy()
    )[:, 1]
    data["regime"] = regime_model.predict(
        data[record.regime_feature_columns].fillna(0.0).to_numpy()
    )
    data["ensemble_score"] = ensemble_signal_score(
        data, data["probability_long"].to_numpy(), data["regime"].to_numpy()
    )
    data["confidence"] = (data["probability_long"] - 0.5).abs() * 2.0

    metadata = metadata_by_root(load_futures_metadata(settings.futures_metadata_path))
    broker = PaperBroker(metadata)
    risk_engine = RiskEngine(
        metadata_by_symbol=metadata,
        config=RiskConfig(
            max_daily_loss=settings.max_daily_loss,
            max_loss_streak=settings.max_loss_streak,
            max_concurrent_positions=settings.max_concurrent_positions,
        ),
    )
    signal_engine = SignalEngine(metadata)
    backtester = EventDrivenBacktester(
        metadata_by_symbol=metadata,
        broker=broker,
        signal_engine=signal_engine,
        risk_engine=risk_engine,
    )
    result = backtester.run(data)
    backtester.save_reports(result, Path(args.reports_dir))

    summary = pd.DataFrame([asdict(result.metrics)])
    summary.to_csv(Path(args.reports_dir) / "metrics_summary.csv", index=False)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)  # noqa: S301 - trusted local registry artifact


if __name__ == "__main__":
    main()

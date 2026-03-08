"""CLI for scheduled retraining, model registration, promotion, and rollback."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
from common.settings import get_settings
from strategy.features import build_feature_frame

from training.dataset import DatasetBuilder
from training.modeling import (
    ensemble_signal_score,
    predict_probability,
    predict_regime,
    to_serializable_metrics,
    train_baseline_models,
)
from training.registry import (
    get_active_records,
    list_records,
    load_record,
    promote_model,
    register_model,
    rollback_model,
)
from training.synthetic_data import create_sample_dataset
from training.workflow import compute_feature_drift, evaluate_promotion_gate


def load_sample_bars(sample_dir: Path, timeframe: str = "1m") -> pd.DataFrame:
    """Load all sample bar files for a given timeframe."""

    files = sorted(sample_dir.glob(f"*_{timeframe}.parquet"))
    if not files:
        files = sorted(sample_dir.glob(f"*_{timeframe}.csv"))
    if not files:
        raise FileNotFoundError(f"No sample bars found in {sample_dir} for timeframe={timeframe}")
    frames = []
    for file in files:
        if file.suffix == ".csv":
            frame = pd.read_csv(file)
        else:
            frame = pd.read_parquet(file)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frames.append(frame)
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["symbol_root", "timestamp"])
        .reset_index(drop=True)
    )


def run_training(output_reports: Path, featureset_version: str = "v1") -> None:
    """Train shadow model and register artifacts locally."""

    settings = get_settings()
    sample_dir = settings.data_dir / "sample"
    registry_dir = settings.model_registry_dir
    output_reports.mkdir(parents=True, exist_ok=True)
    if not sample_dir.exists() or not list(sample_dir.glob("*_1m.*")):
        create_sample_dataset(sample_dir)

    bars = load_sample_bars(sample_dir, timeframe="1m")
    feature_df = build_feature_frame(bars)
    dataset_builder = DatasetBuilder()
    dataset = dataset_builder.build(feature_df)
    train_df, validation_df, test_df = dataset_builder.split_last(dataset)

    trained = train_baseline_models(train_df=train_df, validation_df=validation_df)
    test_prob = predict_probability(trained.model, test_df)
    test_regime = predict_regime(trained.regime_model, test_df)
    test_score = ensemble_signal_score(test_df, probability=test_prob, regime=test_regime)

    test_report = test_df[["timestamp", "symbol_root", "close"]].copy()
    test_report["prob_long"] = test_prob
    test_report["regime"] = test_regime
    test_report["ensemble_score"] = test_score
    test_report.to_csv(output_reports / "recent_predictions.csv", index=False)

    metrics = to_serializable_metrics(trained.metrics)
    training_window = f"{dataset['timestamp'].min()} to {dataset['timestamp'].max()}"
    record = register_model(
        registry_dir=registry_dir,
        model_obj=trained.model,
        regime_model_obj=trained.regime_model,
        training_window=training_window,
        featureset_version=featureset_version,
        metrics=metrics,
        feature_columns=trained.feature_columns,
        regime_feature_columns=trained.regime_feature_columns,
        status="shadow",
    )

    (output_reports / "latest_shadow_model.json").write_text(
        json.dumps(asdict(record), indent=2), encoding="utf-8"
    )

    active = get_active_records(registry_dir)
    production = active["production"]
    promotion_decision = {"promote": False, "reason": "no production comparison"}
    if production is not None:
        production_record = load_record(registry_dir, production.model_id, production.version)
        production_model = _load_pickle(Path(production_record.artifact_path))
        ref_prob = production_model.predict_proba(
            test_df[production_record.feature_columns].fillna(0.0).to_numpy()
        )[:, 1]
        reference = test_df.copy()
        reference["prob"] = ref_prob
        current = test_df.copy()
        current["prob"] = test_prob
        drift = compute_feature_drift(reference, current, trained.feature_columns[:10])
        allowed, diagnostics = evaluate_promotion_gate(
            shadow_metrics=metrics,
            production_metrics=production_record.metrics,
            drift_metrics=drift,
        )
        promotion_decision = {"promote": allowed, "diagnostics": diagnostics}

    (output_reports / "promotion_gate.json").write_text(
        json.dumps(promotion_decision, indent=2), encoding="utf-8"
    )


def cmd_promote(model_id: str, version: str) -> None:
    settings = get_settings()
    promote_model(settings.model_registry_dir, model_id=model_id, version=version)


def cmd_rollback(model_id: str, version: str) -> None:
    settings = get_settings()
    rollback_model(settings.model_registry_dir, model_id=model_id, version=version)


def cmd_list() -> None:
    settings = get_settings()
    rows = [asdict(record) for record in list_records(settings.model_registry_dir)]
    print(json.dumps(rows, indent=2))


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)  # noqa: S301 - trusted local registry artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Model training and registry controls")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Run scheduled retraining and register as shadow")
    train.add_argument("--reports-dir", default="reports", type=str)
    train.add_argument("--featureset-version", default="v1", type=str)

    promote = sub.add_parser("promote", help="Promote a registered model to production")
    promote.add_argument("--model-id", required=True, type=str)
    promote.add_argument("--version", required=True, type=str)

    rollback = sub.add_parser("rollback", help="Rollback to an older model version")
    rollback.add_argument("--model-id", required=True, type=str)
    rollback.add_argument("--version", required=True, type=str)

    sub.add_parser("list", help="List model registry entries")
    args = parser.parse_args()

    if args.command == "train":
        run_training(Path(args.reports_dir), featureset_version=args.featureset_version)
    elif args.command == "promote":
        cmd_promote(args.model_id, args.version)
    elif args.command == "rollback":
        cmd_rollback(args.model_id, args.version)
    elif args.command == "list":
        cmd_list()


if __name__ == "__main__":
    main()

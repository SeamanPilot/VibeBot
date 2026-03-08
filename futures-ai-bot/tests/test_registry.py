from pathlib import Path

from training.registry import list_records, promote_model, register_model


def test_registry_register_and_promote(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    record = register_model(
        registry_dir=registry,
        model_obj={"model": "fake"},
        regime_model_obj={"regime": "fake"},
        training_window="2025-01-01 to 2025-01-10",
        featureset_version="v1",
        metrics={"roc_auc": 0.6},
        feature_columns=["ret_1"],
        regime_feature_columns=["rolling_vol"],
        status="shadow",
    )
    promote_model(registry, record.model_id, record.version)
    rows = list_records(registry)
    assert len(rows) == 1
    assert rows[0].status == "production"

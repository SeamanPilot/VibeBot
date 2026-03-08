"""Local model registry with explicit promotion and rollback controls."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class RegistryRecord:
    """Model registry metadata record."""

    model_id: str
    version: str
    trained_at: str
    training_window: str
    featureset_version: str
    metrics: dict[str, float]
    status: str
    artifact_path: str
    regime_artifact_path: str
    feature_columns: list[str]
    regime_feature_columns: list[str]


def register_model(
    registry_dir: Path,
    model_obj: Any,
    regime_model_obj: Any,
    training_window: str,
    featureset_version: str,
    metrics: dict[str, float],
    feature_columns: list[str],
    regime_feature_columns: list[str],
    status: str = "shadow",
) -> RegistryRecord:
    """Store model artifacts and metadata as a new registry entry."""

    model_id = f"mdl_{uuid4().hex[:8]}"
    version = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    root = registry_dir / model_id / version
    root.mkdir(parents=True, exist_ok=True)

    model_path = root / "model.pkl"
    regime_path = root / "regime_model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model_obj, f)
    with regime_path.open("wb") as f:
        pickle.dump(regime_model_obj, f)

    record = RegistryRecord(
        model_id=model_id,
        version=version,
        trained_at=datetime.now(UTC).isoformat(),
        training_window=training_window,
        featureset_version=featureset_version,
        metrics=metrics,
        status=status,
        artifact_path=str(model_path),
        regime_artifact_path=str(regime_path),
        feature_columns=feature_columns,
        regime_feature_columns=regime_feature_columns,
    )
    _write_record(root / "metadata.json", record)
    _upsert_index(registry_dir, record)
    return record


def list_records(registry_dir: Path) -> list[RegistryRecord]:
    """Read index records sorted by train time descending."""

    index_path = registry_dir / "index.json"
    if not index_path.exists():
        return []
    raw = json.loads(index_path.read_text(encoding="utf-8"))
    items = [RegistryRecord(**row) for row in raw.get("records", [])]
    return sorted(items, key=lambda r: r.trained_at, reverse=True)


def load_record(registry_dir: Path, model_id: str, version: str) -> RegistryRecord:
    """Load a specific record from metadata file."""

    path = registry_dir / model_id / version / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Model metadata not found: {path}")
    return RegistryRecord(**json.loads(path.read_text(encoding="utf-8")))


def promote_model(registry_dir: Path, model_id: str, version: str) -> RegistryRecord:
    """Promote target model to production and archive old production entries."""

    records = list_records(registry_dir)
    promoted: RegistryRecord | None = None
    for rec in records:
        if rec.model_id == model_id and rec.version == version:
            rec.status = "production"
            promoted = rec
        elif rec.status == "production":
            rec.status = "archived"
    if promoted is None:
        raise ValueError(f"Model not found: {model_id}/{version}")
    _write_index(registry_dir, records)
    _persist_record_states(registry_dir, records)
    return promoted


def rollback_model(registry_dir: Path, model_id: str, version: str) -> RegistryRecord:
    """Rollback by promoting a previously registered model version."""

    return promote_model(registry_dir=registry_dir, model_id=model_id, version=version)


def get_active_records(registry_dir: Path) -> dict[str, RegistryRecord | None]:
    """Return active production and newest shadow models."""

    records = list_records(registry_dir)
    production = next((rec for rec in records if rec.status == "production"), None)
    shadow = next((rec for rec in records if rec.status == "shadow"), None)
    return {"production": production, "shadow": shadow}


def _upsert_index(registry_dir: Path, new_record: RegistryRecord) -> None:
    records = list_records(registry_dir)
    records.append(new_record)
    _write_index(registry_dir, records)


def _write_index(registry_dir: Path, records: list[RegistryRecord]) -> None:
    registry_dir.mkdir(parents=True, exist_ok=True)
    index_payload = {"records": [asdict(item) for item in records]}
    (registry_dir / "index.json").write_text(json.dumps(index_payload, indent=2), encoding="utf-8")


def _write_record(path: Path, record: RegistryRecord) -> None:
    path.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")


def _persist_record_states(registry_dir: Path, records: list[RegistryRecord]) -> None:
    for rec in records:
        path = registry_dir / rec.model_id / rec.version / "metadata.json"
        if not path.exists():
            continue
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing["status"] = rec.status
        path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

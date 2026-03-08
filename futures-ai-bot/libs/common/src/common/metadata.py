"""Utilities to load and validate futures metadata YAML."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from common.models import FuturesSymbolMetadata


class FuturesMetadataConfig(BaseModel):
    """Container for all configured symbols."""

    symbols: list[FuturesSymbolMetadata]


def load_futures_metadata(path: Path) -> FuturesMetadataConfig:
    """Load and validate metadata YAML."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return FuturesMetadataConfig.model_validate(raw)


def metadata_by_root(config: FuturesMetadataConfig) -> dict[str, FuturesSymbolMetadata]:
    """Create quick lookup map by symbol root."""

    return {item.root: item for item in config.symbols}

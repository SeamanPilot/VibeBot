from pathlib import Path

from common.metadata import load_futures_metadata


def test_metadata_contains_required_roots() -> None:
    config = load_futures_metadata(Path("config/futures_metadata.yaml"))
    roots = {item.root for item in config.symbols}
    assert roots == {"ES", "NQ", "CL", "GC", "MES", "MNQ", "MCL", "MGC"}

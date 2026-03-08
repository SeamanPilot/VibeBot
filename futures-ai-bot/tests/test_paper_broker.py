from datetime import UTC, datetime
from pathlib import Path

from common.metadata import load_futures_metadata, metadata_by_root
from common.models import NormalizedBar
from executor.models import BracketRequest, OrderRequest, OrderSide, OrderType
from executor.paper_broker import PaperBroker


def _build_broker() -> PaperBroker:
    config = load_futures_metadata(Path("config/futures_metadata.yaml"))
    return PaperBroker(metadata_by_root(config))


def test_market_order_fills_and_opens_position() -> None:
    broker = _build_broker()
    order = broker.submit_order(
        OrderRequest(symbol="ES", side=OrderSide.BUY, quantity=1, order_type=OrderType.MARKET)
    )
    bar = NormalizedBar(
        timestamp=datetime(2025, 1, 1, 14, 30, tzinfo=UTC),
        symbol_root="ES",
        timeframe="1m",
        open=5100.0,
        high=5101.0,
        low=5099.0,
        close=5100.5,
        volume=1000,
        source="test",
    )
    fills = broker.process_bar(bar)
    assert order.order_id == fills[0].order_id
    assert broker.positions["ES"].quantity == 1
    assert broker.orders[order.order_id].status.value == "FILLED"


def test_bracket_order_target_hit_closes_position_and_cancels_stop() -> None:
    broker = _build_broker()
    entry = broker.submit_bracket_order(
        BracketRequest(
            symbol="MES",
            entry_side=OrderSide.BUY,
            quantity=1,
            stop_loss_price=5000.0,
            profit_target_price=5010.0,
        )
    )
    bar_entry = NormalizedBar(
        timestamp=datetime(2025, 1, 1, 14, 30, tzinfo=UTC),
        symbol_root="MES",
        timeframe="1m",
        open=5005.0,
        high=5006.0,
        low=5004.5,
        close=5005.5,
        volume=900,
        source="test",
    )
    broker.process_bar(bar_entry)
    bar_target = NormalizedBar(
        timestamp=datetime(2025, 1, 1, 14, 31, tzinfo=UTC),
        symbol_root="MES",
        timeframe="1m",
        open=5007.0,
        high=5011.5,
        low=5006.0,
        close=5010.5,
        volume=1200,
        source="test",
    )
    broker.process_bar(bar_target)
    assert broker.positions["MES"].quantity == 0
    children = broker.children_by_parent[entry.order_id]
    statuses = {broker.orders[child].status.value for child in children}
    assert "FILLED" in statuses
    assert "CANCELED" in statuses

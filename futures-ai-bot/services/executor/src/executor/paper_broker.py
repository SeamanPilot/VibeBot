"""Event-driven paper broker for futures simulation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime

from common.models import FuturesSymbolMetadata, NormalizedBar

from executor.models import (
    BracketRequest,
    Fill,
    Order,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)


@dataclass(slots=True)
class PaperBrokerConfig:
    """Configurable market simulation parameters."""

    max_fill_ratio_per_bar: float = 1.0
    deterministic_replay: bool = True


class PaperBroker:
    """Simple event-driven paper broker with deterministic replay behavior."""

    def __init__(
        self,
        metadata_by_symbol: dict[str, FuturesSymbolMetadata],
        config: PaperBrokerConfig | None = None,
    ) -> None:
        self.metadata_by_symbol = metadata_by_symbol
        self.config = config or PaperBrokerConfig()

        self.orders: dict[str, Order] = {}
        self.open_orders: list[str] = []
        self.fills: list[Fill] = []
        self.positions: dict[str, Position] = {}
        self.last_price_by_symbol: dict[str, float] = {}
        self.children_by_parent: dict[str, list[str]] = defaultdict(list)

        self.realized_pnl: float = 0.0
        self.unrealized_pnl: float = 0.0
        self.cash: float = 0.0
        self.equity: float = 0.0
        self.peak_equity: float = 0.0
        self.max_drawdown: float = 0.0

    def submit_order(self, request: OrderRequest) -> Order:
        """Accept a new order and place it into the open order book."""

        if request.symbol not in self.metadata_by_symbol:
            raise ValueError(f"Unknown symbol: {request.symbol}")
        order = Order.from_request(request)
        self.orders[order.order_id] = order
        self.open_orders.append(order.order_id)
        if order.parent_order_id is not None:
            self.children_by_parent[order.parent_order_id].append(order.order_id)
        return order

    def submit_bracket_order(self, request: BracketRequest) -> Order:
        """Submit entry order and attach stop/target child orders."""

        entry = self.submit_order(
            OrderRequest(
                symbol=request.symbol,
                side=request.entry_side,
                quantity=request.quantity,
                order_type=request.entry_order_type,
                limit_price=request.entry_limit_price,
                stop_price=request.entry_stop_price,
                metadata={"is_entry": True},
            )
        )

        exit_side = OrderSide.SELL if request.entry_side == OrderSide.BUY else OrderSide.BUY
        oco_group_id = f"oco_{entry.order_id}"
        stop_order = self.submit_order(
            OrderRequest(
                symbol=request.symbol,
                side=exit_side,
                quantity=request.quantity,
                order_type=OrderType.STOP,
                stop_price=request.stop_loss_price,
                parent_order_id=entry.order_id,
                oco_group_id=oco_group_id,
                trailing_ticks=request.trailing_stop_ticks,
                metadata={"is_protective_stop": True},
            )
        )
        limit_order = self.submit_order(
            OrderRequest(
                symbol=request.symbol,
                side=exit_side,
                quantity=request.quantity,
                order_type=OrderType.LIMIT,
                limit_price=request.profit_target_price,
                parent_order_id=entry.order_id,
                oco_group_id=oco_group_id,
                metadata={"is_profit_target": True},
            )
        )
        stop_order.status = OrderStatus.NEW
        limit_order.status = OrderStatus.NEW
        return entry

    def process_bar(self, bar: NormalizedBar) -> list[Fill]:
        """Process one bar and try to execute eligible orders for that symbol."""

        self.last_price_by_symbol[bar.symbol_root] = bar.close
        self._update_trailing_orders(bar)
        fills: list[Fill] = []
        for order_id in list(self.open_orders):
            order = self.orders[order_id]
            if order.symbol != bar.symbol_root:
                continue
            if not self._is_order_active(order):
                continue
            if order.parent_order_id is not None and not self._is_parent_filled(
                order.parent_order_id
            ):
                continue

            should_fill, base_price = self._check_fill(order, bar)
            if not should_fill or base_price is None:
                continue

            fill_qty = self._compute_fill_qty(order)
            if fill_qty <= 0:
                continue

            fill_price = self._apply_slippage(order, base_price)
            fill = self._apply_fill(order=order, bar=bar, quantity=fill_qty, price=fill_price)
            fills.append(fill)
            self.fills.append(fill)
            self._cancel_oco_siblings(order)

        self._mark_to_market()
        return fills

    def cancel_order(self, order_id: str) -> None:
        """Cancel an active order."""

        order = self.orders[order_id]
        if order.status in {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED}:
            return
        order.status = OrderStatus.CANCELED
        if order_id in self.open_orders:
            self.open_orders.remove(order_id)

    def flatten_all(self) -> None:
        """Immediately close all positions at current marked price."""

        synthetic_time = datetime.now(UTC)
        for symbol, position in list(self.positions.items()):
            if position.quantity == 0:
                continue
            last_price = self.last_price_by_symbol.get(symbol)
            if last_price is None:
                continue
            side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
            request = OrderRequest(
                symbol=symbol,
                side=side,
                quantity=abs(position.quantity),
                order_type=OrderType.MARKET,
            )
            order = self.submit_order(request)
            bar = NormalizedBar(
                timestamp=synthetic_time,
                symbol_root=symbol,
                timeframe="1m",
                open=last_price,
                high=last_price,
                low=last_price,
                close=last_price,
                volume=0.0,
                source="flatten",
            )
            self.process_bar(bar)
            self.cancel_order(order.order_id)

    def _check_fill(self, order: Order, bar: NormalizedBar) -> tuple[bool, float | None]:
        if order.order_type == OrderType.MARKET:
            return True, bar.close

        if order.order_type == OrderType.LIMIT and order.limit_price is not None:
            if order.side == OrderSide.BUY and bar.low <= order.limit_price:
                return True, order.limit_price
            if order.side == OrderSide.SELL and bar.high >= order.limit_price:
                return True, order.limit_price
            return False, None

        if order.order_type == OrderType.STOP and order.stop_price is not None:
            if order.side == OrderSide.BUY and bar.high >= order.stop_price:
                return True, order.stop_price
            if order.side == OrderSide.SELL and bar.low <= order.stop_price:
                return True, order.stop_price
            return False, None

        if (
            order.order_type == OrderType.STOP_LIMIT
            and order.stop_price is not None
            and order.limit_price is not None
        ):
            trigger_hit = (order.side == OrderSide.BUY and bar.high >= order.stop_price) or (
                order.side == OrderSide.SELL and bar.low <= order.stop_price
            )
            if not trigger_hit:
                return False, None
            if order.side == OrderSide.BUY and bar.low <= order.limit_price:
                return True, order.limit_price
            if order.side == OrderSide.SELL and bar.high >= order.limit_price:
                return True, order.limit_price
            return False, None

        return False, None

    def _apply_slippage(self, order: Order, reference_price: float) -> float:
        metadata = self.metadata_by_symbol[order.symbol]
        tick = metadata.tick_size
        slip_ticks = metadata.default_slippage_ticks
        if order.order_type == OrderType.LIMIT:
            slip_ticks = max(0.25, slip_ticks * 0.25)
        slip = tick * slip_ticks
        if order.side == OrderSide.BUY:
            return reference_price + slip
        return reference_price - slip

    def _compute_fill_qty(self, order: Order) -> int:
        ratio = min(max(self.config.max_fill_ratio_per_bar, 0.0), 1.0)
        qty = max(int(order.quantity * ratio), 1)
        return min(order.remaining_quantity, qty)

    def _apply_fill(self, order: Order, bar: NormalizedBar, quantity: int, price: float) -> Fill:
        metadata = self.metadata_by_symbol[order.symbol]
        commission = metadata.default_commission_per_contract * quantity
        realized_delta = self._update_position_with_fill(
            symbol=order.symbol, side=order.side, qty=quantity, price=price
        )

        order.remaining_quantity -= quantity
        if order.remaining_quantity == 0:
            order.status = OrderStatus.FILLED
            if order.order_id in self.open_orders:
                self.open_orders.remove(order.order_id)
        else:
            order.status = OrderStatus.PARTIALLY_FILLED

        self.realized_pnl += realized_delta
        self.cash += realized_delta - commission

        fill = Fill(
            timestamp=bar.timestamp,
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=price,
            commission=commission,
            realized_pnl_delta=realized_delta - commission,
        )
        return fill

    def _update_position_with_fill(
        self, symbol: str, side: OrderSide, qty: int, price: float
    ) -> float:
        metadata = self.metadata_by_symbol[symbol]
        tick_size = metadata.tick_size
        tick_value = metadata.tick_value
        signed_qty = qty if side == OrderSide.BUY else -qty

        position = self.positions.get(symbol)
        if position is None:
            position = Position(symbol=symbol)
            self.positions[symbol] = position

        realized_delta = 0.0
        current_qty = position.quantity
        if (
            current_qty == 0
            or (current_qty > 0 and signed_qty > 0)
            or (current_qty < 0 and signed_qty < 0)
        ):
            new_qty = current_qty + signed_qty
            weighted_cost = (abs(current_qty) * position.avg_price) + (qty * price)
            position.avg_price = weighted_cost / abs(new_qty)
            position.quantity = new_qty
            return 0.0

        close_qty = min(abs(current_qty), qty)
        if current_qty > 0 and side == OrderSide.SELL:
            points = price - position.avg_price
            realized_delta = (points / tick_size) * tick_value * close_qty
        elif current_qty < 0 and side == OrderSide.BUY:
            points = position.avg_price - price
            realized_delta = (points / tick_size) * tick_value * close_qty

        remaining_after_close = abs(current_qty) - close_qty
        if remaining_after_close == 0:
            position.quantity = 0
            if qty > close_qty:
                residual_qty = qty - close_qty
                position.quantity = residual_qty if side == OrderSide.BUY else -residual_qty
                position.avg_price = price
            else:
                position.avg_price = 0.0
        else:
            sign = 1 if current_qty > 0 else -1
            position.quantity = sign * remaining_after_close

        position.realized_pnl += realized_delta
        return realized_delta

    def _mark_to_market(self) -> None:
        unrealized = 0.0
        for symbol, position in self.positions.items():
            if position.quantity == 0:
                position.unrealized_pnl = 0.0
                continue
            last_price = self.last_price_by_symbol.get(symbol)
            if last_price is None:
                continue
            metadata = self.metadata_by_symbol[symbol]
            diff = last_price - position.avg_price
            if position.quantity < 0:
                diff = -diff
            pnl = (diff / metadata.tick_size) * metadata.tick_value * abs(position.quantity)
            position.unrealized_pnl = pnl
            unrealized += pnl
        self.unrealized_pnl = unrealized
        self.equity = self.cash + self.unrealized_pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        self.max_drawdown = max(self.max_drawdown, self.peak_equity - self.equity)

    def _update_trailing_orders(self, bar: NormalizedBar) -> None:
        metadata = self.metadata_by_symbol.get(bar.symbol_root)
        if metadata is None:
            return
        tick_size = metadata.tick_size
        for order_id in list(self.open_orders):
            order = self.orders[order_id]
            if (
                order.symbol != bar.symbol_root
                or order.order_type != OrderType.STOP
                or order.trailing_ticks is None
                or order.stop_price is None
            ):
                continue
            offset = order.trailing_ticks * tick_size
            if order.side == OrderSide.SELL:
                new_stop = max(order.stop_price, bar.high - offset)
            else:
                new_stop = min(order.stop_price, bar.low + offset)
            order.stop_price = new_stop

    def _is_parent_filled(self, parent_order_id: str) -> bool:
        return self.orders[parent_order_id].status == OrderStatus.FILLED

    def _is_order_active(self, order: Order) -> bool:
        return order.status in {OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED}

    def _cancel_oco_siblings(self, order: Order) -> None:
        if order.oco_group_id is None or order.status != OrderStatus.FILLED:
            return
        for candidate_id in list(self.open_orders):
            if candidate_id == order.order_id:
                continue
            candidate = self.orders[candidate_id]
            if candidate.oco_group_id == order.oco_group_id:
                self.cancel_order(candidate_id)

    def snapshot(self) -> dict:
        """Return serializable broker state for API and dashboard usage."""

        return {
            "open_orders": [self.orders[oid].model_dump(mode="json") for oid in self.open_orders],
            "fills": [item.model_dump(mode="json") for item in self.fills[-50:]],
            "positions": [pos.model_dump(mode="json") for pos in self.positions.values()],
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "equity": self.equity,
            "max_drawdown": self.max_drawdown,
        }

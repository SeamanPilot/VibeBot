"""Typed order, fill, and position models for paper execution."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class OrderType(StrEnum):
    """Supported paper order types."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(StrEnum):
    """Order side."""

    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(StrEnum):
    """Lifecycle state for an order."""

    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"


class TimeInForce(StrEnum):
    """Time-in-force options."""

    GTC = "GTC"
    IOC = "IOC"


class OrderRequest(BaseModel):
    """Request model for submitting an order."""

    symbol: str
    side: OrderSide
    quantity: int = Field(gt=0)
    order_type: OrderType
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.GTC
    parent_order_id: str | None = None
    oco_group_id: str | None = None
    trailing_ticks: float | None = Field(default=None, ge=0)
    metadata: dict[str, str | float | int | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_prices(self) -> OrderRequest:
        if self.order_type in {OrderType.LIMIT, OrderType.STOP_LIMIT} and self.limit_price is None:
            raise ValueError("limit_price is required for LIMIT and STOP_LIMIT")
        if self.order_type in {OrderType.STOP, OrderType.STOP_LIMIT} and self.stop_price is None:
            raise ValueError("stop_price is required for STOP and STOP_LIMIT")
        return self


class Order(BaseModel):
    """Internal order record."""

    order_id: str = Field(default_factory=lambda: f"ord_{uuid4().hex[:12]}")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    symbol: str
    side: OrderSide
    quantity: int = Field(gt=0)
    remaining_quantity: int = Field(gt=0)
    order_type: OrderType
    status: OrderStatus = OrderStatus.NEW
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.GTC
    parent_order_id: str | None = None
    oco_group_id: str | None = None
    trailing_ticks: float | None = None
    metadata: dict[str, str | float | int | bool] = Field(default_factory=dict)

    @classmethod
    def from_request(cls, request: OrderRequest) -> Order:
        """Build internal order from request."""

        return cls(
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            remaining_quantity=request.quantity,
            order_type=request.order_type,
            limit_price=request.limit_price,
            stop_price=request.stop_price,
            time_in_force=request.time_in_force,
            parent_order_id=request.parent_order_id,
            oco_group_id=request.oco_group_id,
            trailing_ticks=request.trailing_ticks,
            metadata=request.metadata,
        )


class Fill(BaseModel):
    """Fill record emitted by the paper broker."""

    fill_id: str = Field(default_factory=lambda: f"fill_{uuid4().hex[:12]}")
    timestamp: datetime
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    commission: float = Field(ge=0)
    realized_pnl_delta: float = 0.0


class Position(BaseModel):
    """Open position state for a symbol."""

    symbol: str
    quantity: int = 0
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0


class BracketRequest(BaseModel):
    """Bracket order request for one entry plus attached exits."""

    symbol: str
    entry_side: OrderSide
    quantity: int = Field(gt=0)
    entry_order_type: OrderType = OrderType.MARKET
    entry_limit_price: float | None = None
    entry_stop_price: float | None = None
    stop_loss_price: float
    profit_target_price: float
    trailing_stop_ticks: float | None = Field(default=None, ge=0)

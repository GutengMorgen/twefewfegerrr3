from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace
from typing import Literal, Protocol

import pandas as pd

from trading.account.inventory import AccountInventory, AccountPosition
from trading.helpers.market import MarketSpec


Side = Literal["buy", "sell"]
OrderAction = Literal["open", "close"]


class TickLike(Protocol):
    timestamp: pd.Timestamp
    bid: float
    ask: float
    mid: float


@dataclass(frozen=True)
class ExecutionCostModel:
    commission_per_order: float = 0.0
    fee_per_order: float = 0.0
    slippage_points: float = 0.0
    latency_ms: int = 0


@dataclass(frozen=True)
class ExecutionSettings:
    cost_model: ExecutionCostModel = field(default_factory=lambda: DEFAULT_EXECUTION_COSTS)
    default_order_size: float = 0.02


DEFAULT_EXECUTION_COSTS = ExecutionCostModel(
    commission_per_order=0.35,
    fee_per_order=0.05,
    slippage_points=0.25,
    latency_ms=150,
)


@dataclass(frozen=True)
class OrderRequest:
    action: OrderAction
    side: Side
    epic: str
    size: float
    submitted_at: pd.Timestamp
    reference_price: float


@dataclass(frozen=True)
class PendingOrder:
    order_id: int
    request: OrderRequest
    eligible_at: pd.Timestamp


@dataclass(frozen=True)
class ExecutionReceipt:
    order_id: int
    action: OrderAction
    side: Side
    epic: str
    requested_at: pd.Timestamp
    filled_at: pd.Timestamp
    requested_price: float
    fill_price: float
    commission: float
    fee: float
    spread: float
    slippage_points: float
    latency_ms: int


class SimulatedExecutionEngine:
    def __init__(self, settings: ExecutionSettings | None = None) -> None:
        self.settings = settings or ExecutionSettings()
        self.cost_model = self.settings.cost_model
        self._pending_orders: list[PendingOrder] = []
        self._next_order_id = 1

    @staticmethod
    def _normalize_size(size: float) -> float:
        return round(float(size), 2)

    @property
    def pending_count(self) -> int:
        return len(self._pending_orders)

    def submit_order(self, request: OrderRequest) -> PendingOrder:
        normalized_request = replace(request, size=self._normalize_size(request.size))
        latency = pd.Timedelta(milliseconds=max(self.cost_model.latency_ms, 0))
        pending_order = PendingOrder(
            order_id=self._next_order_id,
            request=normalized_request,
            eligible_at=normalized_request.submitted_at + latency,
        )
        self._pending_orders.append(pending_order)
        self._next_order_id += 1
        return pending_order

    def process_pending(
        self,
        *,
        tick: TickLike,
        wallet: AccountInventory,
        market_spec: MarketSpec,
    ) -> list[ExecutionReceipt]:
        ready_orders = [order for order in self._pending_orders if order.eligible_at <= tick.timestamp]
        self._pending_orders = [order for order in self._pending_orders if order.eligible_at > tick.timestamp]

        receipts: list[ExecutionReceipt] = []
        for order in ready_orders:
            receipts.append(self._fill_order(order, tick, wallet, market_spec))
        return receipts

    def _fill_order(
        self,
        order: PendingOrder,
        tick: TickLike,
        wallet: AccountInventory,
        market_spec: MarketSpec,
    ) -> ExecutionReceipt:
        slippage = max(self.cost_model.slippage_points, 0.0)
        spread = max(tick.ask - tick.bid, 0.0)
        if order.request.action == "open":
            fill_price = tick.ask + slippage if order.request.side == "buy" else tick.bid - slippage
            position = AccountPosition(
                epic=order.request.epic,
                side=order.request.side,
                size=order.request.size,
                entry_price=fill_price,
                contract_size=market_spec.contract_size,
                margin_factor=market_spec.margin_factor,
                opened_at=tick.timestamp,
            )
            receipt = ExecutionReceipt(
                order_id=order.order_id,
                action="open",
                side=order.request.side,
                epic=order.request.epic,
                requested_at=order.request.submitted_at,
                filled_at=tick.timestamp,
                requested_price=order.request.reference_price,
                fill_price=fill_price,
                commission=self.cost_model.commission_per_order,
                fee=self.cost_model.fee_per_order,
                spread=spread,
                slippage_points=slippage,
                latency_ms=max(self.cost_model.latency_ms, 0),
            )
            wallet.open_position(position, receipt=receipt)
            return receipt

        position = wallet.position_for(order.request.epic)
        if position is None:
            raise RuntimeError(f"Cannot close {order.request.epic!r}; no open position is available")

        expected_close_side: Side = "sell" if position.side == "buy" else "buy"
        if order.request.side != expected_close_side:
            raise ValueError(
                f"Close request side {order.request.side!r} does not match the open position side {position.side!r}"
            )

        fill_price = tick.bid - slippage if order.request.side == "sell" else tick.ask + slippage
        receipt = ExecutionReceipt(
            order_id=order.order_id,
            action="close",
            side=order.request.side,
            epic=order.request.epic,
            requested_at=order.request.submitted_at,
            filled_at=tick.timestamp,
            requested_price=order.request.reference_price,
            fill_price=fill_price,
            commission=self.cost_model.commission_per_order,
            fee=self.cost_model.fee_per_order,
            spread=spread,
            slippage_points=slippage,
            latency_ms=max(self.cost_model.latency_ms, 0),
        )
        wallet.close_position(order.request.epic, fill_price, receipt=receipt)
        return receipt

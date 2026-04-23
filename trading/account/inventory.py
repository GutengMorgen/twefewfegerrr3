from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal

import pandas as pd


Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class AccountSettings:
    initial_cash: float = 50.0


@dataclass
class AccountPosition:
    epic: str
    side: Side
    size: float
    entry_price: float
    contract_size: float
    margin_factor: float
    opened_at: pd.Timestamp | None = None

    def margin_required(self) -> float:
        exposure = self.entry_price * self.contract_size * self.size
        return exposure * (self.margin_factor / 100.0)

    def unrealized_pnl(self, current_price: float) -> float:
        direction = 1.0 if self.side == "buy" else -1.0
        return (current_price - self.entry_price) * self.contract_size * self.size * direction


def _build_operation_log_entry(
    *,
    operation: str,
    position: AccountPosition,
    cash_after: float,
    reserved_margin_after: float,
    realized_pnl_after: float,
    receipt: object | None,
    fill_price: float,
    margin_required: float,
    realized_pnl: float = 0.0,
) -> dict[str, object]:
    return {
        "operation": operation,
        "epic": position.epic,
        "side": position.side,
        "size": position.size,
        "entry_price": position.entry_price,
        "fill_price": fill_price,
        "margin_required": margin_required,
        "commission": 0.0 if receipt is None else float(getattr(receipt, "commission", 0.0)),
        "fee": 0.0 if receipt is None else float(getattr(receipt, "fee", 0.0)),
        "spread": 0.0 if receipt is None else float(getattr(receipt, "spread", 0.0)),
        "slippage_points": 0.0 if receipt is None else float(getattr(receipt, "slippage_points", 0.0)),
        "latency_ms": 0 if receipt is None else int(getattr(receipt, "latency_ms", 0)),
        "requested_at": None if receipt is None else getattr(receipt, "requested_at", None),
        "filled_at": None if receipt is None else getattr(receipt, "filled_at", None),
        "requested_price": position.entry_price if receipt is None else getattr(receipt, "requested_price", position.entry_price),
        "realized_pnl": realized_pnl,
        "cash_after": cash_after,
        "reserved_margin_after": reserved_margin_after,
        "realized_pnl_after": realized_pnl_after,
    }


@dataclass
class AccountInventory:
    settings: AccountSettings = field(default_factory=AccountSettings)
    cash: float = field(init=False)
    reserved_margin: float = 0.0
    realized_pnl: float = 0.0
    positions: list[AccountPosition] = field(default_factory=list)
    operation_log: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cash = float(self.settings.initial_cash)

    @property
    def initial_cash(self) -> float:
        return self.settings.initial_cash

    @property
    def balance(self) -> float:
        return self.cash

    @property
    def equity(self) -> float:
        return self.cash + self.reserved_margin + sum(
            position.unrealized_pnl(position.entry_price) for position in self.positions
        )

    @property
    def trade_log(self) -> list[dict[str, object]]:
        return self.operation_log

    def position_for(self, epic: str) -> AccountPosition | None:
        for position in self.positions:
            if position.epic == epic:
                return position
        return None

    def position_summary(self) -> str:
        if not self.positions:
            return "position=NO_POSITION"
        position = self.positions[0]
        return f"position={position.side.upper()} size={position.size:.2f} entry={position.entry_price:.2f}"

    def open_position(self, position: AccountPosition, receipt: object | None = None) -> None:
        if not math.isfinite(position.entry_price):
            raise ValueError(f"entry_price must be finite; received {position.entry_price!r}")
        if not math.isfinite(position.size) or position.size <= 0.0:
            raise ValueError(f"size must be > 0 and finite; received {position.size!r}")

        margin = position.margin_required()
        total_cost = margin
        if receipt is not None:
            total_cost += float(getattr(receipt, "commission", 0.0)) + float(getattr(receipt, "fee", 0.0))
        if total_cost > self.cash:
            raise ValueError(
                f"Not enough virtual cash to open {position.epic} size {position.size}: "
                f"need {total_cost:.2f}, have {self.cash:.2f}"
            )

        new_cash = self.cash - total_cost
        new_reserved_margin = self.reserved_margin + margin

        self.cash = new_cash
        self.reserved_margin = new_reserved_margin
        self.positions.append(position)
        self.operation_log.append(
            _build_operation_log_entry(
                operation="open",
                position=position,
                cash_after=self.cash,
                reserved_margin_after=self.reserved_margin,
                realized_pnl_after=self.realized_pnl,
                receipt=receipt,
                fill_price=position.entry_price if receipt is None else float(getattr(receipt, "fill_price", position.entry_price)),
                margin_required=margin,
            )
        )

    def close_position(self, epic: str, exit_price: float, receipt: object | None = None) -> AccountPosition:
        if not math.isfinite(exit_price):
            raise ValueError(f"exit_price must be finite; received {exit_price!r}")

        for index, position in enumerate(self.positions):
            if position.epic == epic:
                pnl = position.unrealized_pnl(exit_price)
                margin = position.margin_required()
                total_cost = 0.0 if receipt is None else float(getattr(receipt, "commission", 0.0)) + float(getattr(receipt, "fee", 0.0))

                new_cash = self.cash + margin + pnl - total_cost
                new_reserved_margin = self.reserved_margin - margin
                new_realized_pnl = self.realized_pnl + pnl
                if new_reserved_margin < -1e-9:
                    raise RuntimeError(
                        f"Invalid reserved margin after closing {epic!r}: {new_reserved_margin:.6f}"
                    )

                self.positions.pop(index)
                self.cash = new_cash
                self.reserved_margin = max(new_reserved_margin, 0.0)
                self.realized_pnl = new_realized_pnl
                self.operation_log.append(
                    _build_operation_log_entry(
                        operation="close",
                        position=position,
                        cash_after=self.cash,
                        reserved_margin_after=self.reserved_margin,
                        realized_pnl_after=self.realized_pnl,
                        receipt=receipt,
                        fill_price=exit_price,
                        margin_required=margin,
                        realized_pnl=pnl,
                    )
                )
                return position
        raise ValueError(f"No open virtual position found for {epic!r}")

    def snapshot(self, current_price: float | None = None) -> dict[str, float]:
        unrealized = 0.0
        if current_price is not None:
            if not math.isfinite(current_price):
                raise ValueError(f"current_price must be finite; received {current_price!r}")
            unrealized = sum(position.unrealized_pnl(current_price) for position in self.positions)
        return {
            "cash": self.cash,
            "balance": self.cash,
            "reserved_margin": self.reserved_margin,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": unrealized,
            "equity": self.cash + self.reserved_margin + unrealized,
        }

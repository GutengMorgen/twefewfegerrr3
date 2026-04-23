from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


@dataclass(frozen=True)
class TerminalDisplayConfig:
    show_timestamp: bool = True
    show_price: bool = True
    show_spread: bool = True
    show_market: bool = True
    show_stream_status: bool = True
    show_strategy_status: bool = True
    show_execution_status: bool = True
    show_positions: bool = True
    show_unrealized_pnl: bool = True
    show_equity: bool = True
    show_signal_criteria: bool = True
    show_market_metadata: bool = True
    show_account_metadata: bool = True
    show_trade_extremes: bool = True
    show_volume: bool = True
    show_bar_ohlc: bool = True
    max_criteria_rows: int = 8


class RichLiveTerminal:
    def __init__(self, display: TerminalDisplayConfig | None = None, *, title: str = "Virtual Wallet") -> None:
        self.display = display or TerminalDisplayConfig()
        self.title = title
        self.console = Console()
        self._live: Live | None = None
        self._last_snapshot: dict[str, Any] | None = None

    def __enter__(self) -> "RichLiveTerminal":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        if self._live is not None:
            return

        self._live = Live(
            self._render(self._last_snapshot),
            console=self.console,
            refresh_per_second=4,
            screen=False,
            transient=False,
        )
        self._live.__enter__()

    def stop(self) -> None:
        if self._live is None:
            return

        self._live.__exit__(None, None, None)
        self._live = None

    def message(self, text: str, *, style: str = "yellow") -> None:
        self.console.print(text, style=style)

    def update(self, snapshot: dict[str, Any]) -> None:
        self._last_snapshot = snapshot
        renderable = self._render(snapshot)
        if self._live is None:
            self.console.print(renderable)
            return
        self._live.update(renderable)

    def _render(self, snapshot: dict[str, Any] | None) -> Group:
        if not snapshot:
            return Group(Panel("Waiting for live data...", title=self.title, box=box.ROUNDED))

        summary_panel = Panel(self._render_summary_table(snapshot), title=self.title, box=box.ROUNDED)
        criteria_panel = Panel(self._render_criteria_table(snapshot), title="Signal Criteria", box=box.ROUNDED)
        panels = [summary_panel]
        if self.display.show_signal_criteria:
            panels.append(criteria_panel)
        return Group(*panels)

    def _render_summary_table(self, snapshot: dict[str, Any]) -> Table:
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column(style="white")

        market = snapshot.get("market", {}) or {}
        strategy = snapshot.get("strategy", {}) or {}
        execution = snapshot.get("execution", {}) or {}
        account = snapshot.get("account", {}) or {}
        position = snapshot.get("position", {}) or {}
        extrema = snapshot.get("extrema", {}) or {}
        bar = market.get("bar", {}) or {}

        if self.display.show_timestamp:
            timestamp = snapshot.get("timestamp")
            table.add_row("time", self._format_timestamp(timestamp))

        if self.display.show_market_metadata:
            if self.display.show_market:
                market_name = market.get("name") or market.get("epic") or "-"
                table.add_row("market", str(market_name))

            if self.display.show_price:
                price_parts = [
                    f"bid={self._format_number(market.get('bid'), 2)}",
                    f"ask={self._format_number(market.get('ask'), 2)}",
                    f"mid={self._format_number(market.get('mid'), 2)}",
                ]
                if self.display.show_spread:
                    price_parts.append(f"spread={self._format_number(market.get('spread'), 2)}")
                if self.display.show_volume:
                    price_parts.append(f"vol={self._format_number(market.get('volume'))}")
                table.add_row("price", " | ".join(price_parts))

            if self.display.show_bar_ohlc:
                bar_parts = [
                    f"open={self._format_number(bar.get('open'), 2)}",
                    f"high={self._format_number(bar.get('high'), 2)}",
                    f"low={self._format_number(bar.get('low'), 2)}",
                    f"close={self._format_number(bar.get('close'), 2)}",
                ]
                if self.display.show_volume:
                    bar_parts.append(f"bar_vol={self._format_number(bar.get('volume'))}")
                table.add_row("bar", " | ".join(bar_parts))

        if self.display.show_strategy_status:
            strategy_parts = [
                f"name={strategy.get('name', '-')}",
                f"state={strategy.get('state', '-')}",
                f"decision={strategy.get('decision_action', '-')}",
            ]
            if strategy.get("decision_side"):
                strategy_parts.append(f"side={strategy.get('decision_side')}")
            if strategy.get("decision_reason"):
                strategy_parts.append(f"reason={strategy.get('decision_reason')}")
            table.add_row("strategy", " | ".join(strategy_parts))

        if self.display.show_execution_status:
            execution_parts = [
                f"stream={execution.get('stream', '-')}",
                f"strategy={execution.get('strategy', '-')}",
                f"execution={execution.get('execution', '-')}",
                f"pending={execution.get('pending_orders', '-')}",
                f"market={execution.get('market', '-')}",
            ]
            table.add_row("runtime", " | ".join(execution_parts))

        if self.display.show_positions:
            table.add_row("position", str(position.get("summary", "position=NO_POSITION")))

        if self.display.show_account_metadata:
            account_parts = [
                f"cash={self._format_number(account.get('cash'), 2)}",
                f"balance={self._format_number(account.get('balance'), 2)}",
                f"equity={self._format_number(account.get('equity'), 2)}",
            ]
            if self.display.show_unrealized_pnl:
                account_parts.append(f"pnl={self._format_number(account.get('unrealized_pnl'), 2)}")
            if account.get("reserved_margin") is not None:
                account_parts.append(f"margin={self._format_number(account.get('reserved_margin'), 2)}")
            table.add_row("account", " | ".join(account_parts))

        if self.display.show_trade_extremes:
            extrema_parts = [
                f"session_hi_eq={self._format_number(extrema.get('session_high_equity'), 2)}",
                f"session_lo_eq={self._format_number(extrema.get('session_low_equity'), 2)}",
                f"session_hi_px={self._format_number(extrema.get('session_high_price'), 2)}",
                f"session_lo_px={self._format_number(extrema.get('session_low_price'), 2)}",
            ]
            if extrema.get("trade_high_equity") is not None or extrema.get("trade_low_equity") is not None:
                extrema_parts.extend(
                    [
                        f"trade_hi_eq={self._format_number(extrema.get('trade_high_equity'), 2)}",
                        f"trade_lo_eq={self._format_number(extrema.get('trade_low_equity'), 2)}",
                        f"trade_hi_px={self._format_number(extrema.get('trade_high_price'), 2)}",
                        f"trade_lo_px={self._format_number(extrema.get('trade_low_price'), 2)}",
                    ]
                )
            table.add_row("extrema", " | ".join(extrema_parts))

        return table

    def _render_criteria_table(self, snapshot: dict[str, Any]) -> Table:
        table = Table(box=box.MINIMAL, expand=True, show_header=True, header_style="bold magenta")
        table.add_column("kind", style="cyan", no_wrap=True)
        table.add_column("label", style="bold", no_wrap=True)
        table.add_column("current", justify="right")
        table.add_column("target", justify="right")
        table.add_column("cmp", no_wrap=True)
        table.add_column("state", no_wrap=True)
        table.add_column("note")

        strategy = snapshot.get("strategy", {}) or {}
        criteria = list(strategy.get("criteria", []) or [])
        if not criteria:
            table.add_row("-", "no criteria", "-", "-", "-", "-", "")
            return table

        for criterion in criteria[: self.display.max_criteria_rows]:
            active = criterion.get("active")
            active_text = "on" if active else "off" if active is not None else "-"
            active_style = "green" if active else "red" if active is False else "dim"
            table.add_row(
                str(criterion.get("kind", "-")),
                str(criterion.get("label", "-")),
                self._format_criterion_value(criterion.get("current")),
                self._format_criterion_value(criterion.get("target")),
                str(criterion.get("comparator", "-")),
                Text(active_text, style=active_style),
                str(criterion.get("note") or ""),
            )

        extra_rows = len(criteria) - self.display.max_criteria_rows
        if extra_rows > 0:
            table.add_row("-", f"... {extra_rows} more", "-", "-", "-", "-", "")

        return table

    @staticmethod
    def _format_timestamp(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m-%d %H:%M:%S")
        return str(value)

    @staticmethod
    def _format_number(value: Any, precision: int = 4) -> str:
        if value is None:
            return "-"
        if isinstance(value, (int, float)):
            return f"{value:.{precision}f}"
        return str(value)

    @classmethod
    def _format_criterion_value(cls, value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, (int, float)):
            return f"{value:.4f}"
        return str(value)

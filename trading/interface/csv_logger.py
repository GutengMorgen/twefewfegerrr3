from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_CSV_FIELDNAMES = (
    "event_type",
    "timestamp",
    "epic",
    "market_name",
    "market_contract_size",
    "market_margin_factor",
    "bid",
    "ask",
    "mid",
    "spread",
    "volume",
    "bar_open",
    "bar_high",
    "bar_low",
    "bar_close",
    "bar_volume",
    "stream_state",
    "strategy_enabled",
    "execution_enabled",
    "pending_orders",
    "strategy_name",
    "strategy_state",
    "strategy_decision_action",
    "strategy_decision_side",
    "strategy_decision_reason",
    "strategy_criteria_json",
    "strategy_metadata_json",
    "position_summary",
    "position_side",
    "position_size",
    "position_entry_price",
    "position_count",
    "cash",
    "balance",
    "reserved_margin",
    "realized_pnl",
    "unrealized_pnl",
    "equity",
    "session_high_equity",
    "session_low_equity",
    "session_high_price",
    "session_low_price",
    "trade_high_equity",
    "trade_low_equity",
    "trade_high_price",
    "trade_low_price",
    "market_metadata_json",
    "account_metadata_json",
    "position_metadata_json",
    "extrema_metadata_json",
    "extra_json",
    "note",
)


@dataclass
class CsvRunLogger:
    csv_path: Path
    fieldnames: tuple[str, ...] = field(default=DEFAULT_CSV_FIELDNAMES)

    def __post_init__(self) -> None:
        self.csv_path = Path(self.csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    def log_snapshot(
        self,
        snapshot: Mapping[str, Any],
        *,
        event_type: str = "snapshot",
        note: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        row = self._build_row(snapshot, event_type=event_type, note=note, extra=extra)
        self._append_row(row)

    def _append_row(self, row: Mapping[str, Any]) -> None:
        try:
            write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
            with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames, extrasaction="ignore")
                if write_header:
                    writer.writeheader()
                writer.writerow({key: row.get(key, "") for key in self.fieldnames})
        except (OSError, csv.Error) as exc:
            raise RuntimeError(f"Unable to write CSV log at {self.csv_path}: {exc}") from exc

    def _build_row(
        self,
        snapshot: Mapping[str, Any],
        *,
        event_type: str,
        note: str | None,
        extra: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        market = dict(snapshot.get("market", {}) or {})
        strategy = dict(snapshot.get("strategy", {}) or {})
        execution = dict(snapshot.get("execution", {}) or {})
        account = dict(snapshot.get("account", {}) or {})
        position = dict(snapshot.get("position", {}) or {})
        extrema = dict(snapshot.get("extrema", {}) or {})
        bar = dict(market.get("bar", {}) or {})

        row: dict[str, Any] = {
            "event_type": event_type,
            "timestamp": self._stringify(snapshot.get("timestamp")),
            "epic": market.get("epic", ""),
            "market_name": market.get("name", ""),
            "market_contract_size": market.get("contract_size", ""),
            "market_margin_factor": market.get("margin_factor", ""),
            "bid": market.get("bid", ""),
            "ask": market.get("ask", ""),
            "mid": market.get("mid", ""),
            "spread": market.get("spread", ""),
            "volume": market.get("volume", ""),
            "bar_open": bar.get("open", ""),
            "bar_high": bar.get("high", ""),
            "bar_low": bar.get("low", ""),
            "bar_close": bar.get("close", ""),
            "bar_volume": bar.get("volume", ""),
            "stream_state": execution.get("stream", ""),
            "strategy_enabled": execution.get("strategy_enabled", ""),
            "execution_enabled": execution.get("execution_enabled", ""),
            "pending_orders": execution.get("pending_orders", ""),
            "strategy_name": strategy.get("name", ""),
            "strategy_state": strategy.get("state", ""),
            "strategy_decision_action": strategy.get("decision_action", ""),
            "strategy_decision_side": strategy.get("decision_side", ""),
            "strategy_decision_reason": strategy.get("decision_reason", ""),
            "strategy_criteria_json": json.dumps(strategy.get("criteria", []), default=self._stringify, ensure_ascii=False),
            "strategy_metadata_json": json.dumps(strategy.get("metadata", {}), default=self._stringify, ensure_ascii=False),
            "position_summary": position.get("summary", ""),
            "position_side": position.get("side", ""),
            "position_size": position.get("size", ""),
            "position_entry_price": position.get("entry_price", ""),
            "position_count": account.get("position_count", ""),
            "cash": account.get("cash", ""),
            "balance": account.get("balance", ""),
            "reserved_margin": account.get("reserved_margin", ""),
            "realized_pnl": account.get("realized_pnl", ""),
            "unrealized_pnl": account.get("unrealized_pnl", ""),
            "equity": account.get("equity", ""),
            "session_high_equity": extrema.get("session_high_equity", ""),
            "session_low_equity": extrema.get("session_low_equity", ""),
            "session_high_price": extrema.get("session_high_price", ""),
            "session_low_price": extrema.get("session_low_price", ""),
            "trade_high_equity": extrema.get("trade_high_equity", ""),
            "trade_low_equity": extrema.get("trade_low_equity", ""),
            "trade_high_price": extrema.get("trade_high_price", ""),
            "trade_low_price": extrema.get("trade_low_price", ""),
            "market_metadata_json": json.dumps(market, default=self._stringify, ensure_ascii=False),
            "account_metadata_json": json.dumps(account, default=self._stringify, ensure_ascii=False),
            "position_metadata_json": json.dumps(position, default=self._stringify, ensure_ascii=False),
            "extrema_metadata_json": json.dumps(extrema, default=self._stringify, ensure_ascii=False),
            "extra_json": json.dumps(dict(extra or {}), default=self._stringify, ensure_ascii=False),
            "note": note or "",
        }
        return row

    @staticmethod
    def _stringify(value: Any) -> Any:
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)

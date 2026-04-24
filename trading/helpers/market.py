from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from trading.ig_nq_data import get_market_by_epic
from trading_ig.rest import IGException


@dataclass(frozen=True)
class MarketSpec:
    epic: str
    name: str
    contract_size: float
    margin_factor: float


@dataclass(frozen=True)
class MarketTradingState:
    market_status: str
    is_open: bool
    reason: str | None = None


FALLBACK_MARKET_SPECS: dict[str, dict[str, float | str]] = {
    "IX.D.NASDAQ.IFMM.IP": {
        "name": "EEUU Tech 100 al contado (1$)",
        "contractSize": 1.0,
        "marginFactor": 0.5,
    },
    "IX.D.NASDAQ.FBMU1.IP": {
        "name": "EEUU Tech 100 (1$)",
        "contractSize": 1.0,
        "marginFactor": 0.5,
    },
}


def _field_value(source: Any, key: str, default: Any = None) -> Any:
    if hasattr(source, "get"):
        return source.get(key, default)
    return getattr(source, key, default)


def _to_finite_float(value: Any, *, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric; received {value!r}") from exc

    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite; received {value!r}")
    return parsed


def _normalize_market_status(value: Any) -> str:
    status = str(value or "").strip().upper()
    return status or "UNKNOWN"


def market_trading_state(market: Any) -> MarketTradingState:
    snapshot = _field_value(market, "snapshot", {})
    market_status = _normalize_market_status(
        _field_value(
            market,
            "marketStatus",
            _field_value(market, "status", _field_value(snapshot, "marketStatus", _field_value(snapshot, "status", ""))),
        )
    )
    is_open = market_status in {"TRADEABLE", "OPEN", "DEALABLE"}
    reason = None if is_open else f"marketStatus={market_status}"
    return MarketTradingState(market_status=market_status, is_open=is_open, reason=reason)


def resolve_market_spec(epic: str) -> MarketSpec:
    fallback = FALLBACK_MARKET_SPECS.get(epic)
    if fallback is None:
        raise KeyError(f"No fallback market spec is defined for {epic!r}")
    return MarketSpec(
        epic=epic,
        name=str(fallback["name"]),
        contract_size=float(fallback["contractSize"]),
        margin_factor=float(fallback["marginFactor"]),
    )


def load_market_spec(service: Any, epic: str) -> MarketSpec:
    try:
        market = get_market_by_epic(service, epic)
    except IGException:
        return resolve_market_spec(epic)

    instrument = _field_value(market, "instrument", {})
    name = str(_field_value(instrument, "name", epic)).strip() or epic
    contract_size = _to_finite_float(_field_value(instrument, "contractSize", 1.0), field_name="contractSize")
    margin_factor = _to_finite_float(_field_value(instrument, "marginFactor", 0.5), field_name="marginFactor")

    return MarketSpec(
        epic=epic,
        name=name,
        contract_size=contract_size,
        margin_factor=margin_factor,
    )

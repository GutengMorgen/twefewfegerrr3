"""Terminal-only live trading runner for virtual Nasdaq trading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd
from trading_ig.rest import IGException

# Allow running this file directly in addition to module mode.
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from trading.account.inventory import AccountInventory
from trading.interface.csv_logger import CsvRunLogger
from trading.interface.terminal import RichLiveTerminal, TerminalDisplayConfig
from trading.virtual_wallet.execution import (
    ExecutionReceipt,
    OrderRequest,
    Side,
    SimulatedExecutionEngine,
)
from trading.helpers.market import MarketTradingState, load_market_spec, market_trading_state
from trading.streaming.price_feed import LiveBarBuilder, LivePriceFeed, LivePriceFeedConfig
from trading.strategies.base import StrategyContext, StrategyDecision, StrategyLiveSnapshot
from trading.strategies.registry import DEFAULT_STRATEGY_REGISTRY
from trading.ig_nq_data import build_service, get_market_by_epic, load_credentials, load_or_fetch_historical_ohlcv


DEFAULT_EPIC = "IX.D.NASDAQ.IFMM.IP"
DEFAULT_STREAM_PRINT_SECONDS = 1.0
DEFAULT_MARKET_CLOSED_SLEEP_SECONDS = 30.0
DEFAULT_RESTART_BACKOFF_SECONDS = 30.0
DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("virtual_wallet_config.json")
DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "logs" / "virtual_wallet_live.csv"


@dataclass(frozen=True)
class WorkflowLoggingConfig:
    csv_path: Path = DEFAULT_LOG_PATH


@dataclass
class RangeTracker:
    high_equity: float | None = None
    low_equity: float | None = None
    high_price: float | None = None
    low_price: float | None = None
    start_time: pd.Timestamp | None = None
    end_time: pd.Timestamp | None = None
    active: bool = False

    def start(self, *, timestamp: pd.Timestamp, price: float, equity: float) -> None:
        self.active = True
        self.start_time = timestamp
        self.end_time = None
        self.high_equity = equity
        self.low_equity = equity
        self.high_price = price
        self.low_price = price

    def update(self, *, timestamp: pd.Timestamp, price: float, equity: float) -> None:
        if not self.active:
            self.start(timestamp=timestamp, price=price, equity=equity)
            return

        self.end_time = timestamp
        if self.high_equity is None or equity > self.high_equity:
            self.high_equity = equity
        if self.low_equity is None or equity < self.low_equity:
            self.low_equity = equity
        if self.high_price is None or price > self.high_price:
            self.high_price = price
        if self.low_price is None or price < self.low_price:
            self.low_price = price

    def reset(self) -> None:
        self.high_equity = None
        self.low_equity = None
        self.high_price = None
        self.low_price = None
        self.start_time = None
        self.end_time = None
        self.active = False

    def snapshot(self) -> dict[str, float | str | None]:
        return {
            "trade_high_equity": self.high_equity,
            "trade_low_equity": self.low_equity,
            "trade_high_price": self.high_price,
            "trade_low_price": self.low_price,
            "trade_start_time": self.start_time,
            "trade_end_time": self.end_time,
        }


@dataclass(frozen=True)
class WorkflowConfig:
    config_path: Path = DEFAULT_CONFIG_PATH
    enabled: bool = True
    reload_interval_seconds: float = 30.0
    epic: str = DEFAULT_EPIC
    strategy_name: str | None = "std_levels_touch_density"
    strategy_overrides: dict[str, Any] | None = None
    stream_price_enabled: bool = True
    strategy_process_enabled: bool = False
    execution_process_enabled: bool = True
    stream_print_seconds: float = DEFAULT_STREAM_PRINT_SECONDS
    stream: LivePriceFeedConfig = field(default_factory=LivePriceFeedConfig)
    terminal_display: TerminalDisplayConfig = field(default_factory=TerminalDisplayConfig)
    logging: WorkflowLoggingConfig = field(default_factory=WorkflowLoggingConfig)


def _resolve_path(base_dir: Path, configured_value: Any, default_path: Path) -> Path:
    if configured_value is None or str(configured_value).strip() == "":
        path = default_path
    else:
        path = Path(str(configured_value))
        if not path.is_absolute():
            path = base_dir / path
    return path


def _parse_bool(value: Any, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{key} must be a boolean; received {value!r}")


def _parse_float(value: Any, *, key: str, minimum: float | None = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a float; received {value!r}") from exc

    if not math.isfinite(parsed):
        raise ValueError(f"{key} must be finite; received {value!r}")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{key} must be >= {minimum}; received {parsed}")
    return parsed


def _parse_int(value: Any, *, key: str, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer; received {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer; received {value!r}") from exc

    if minimum is not None and parsed < minimum:
        raise ValueError(f"{key} must be >= {minimum}; received {parsed}")
    return parsed


def _coerce_strategy_overrides(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("workflow.strategy_overrides must be a JSON object")
    return dict(value)


def _resolve_order_size(decision: StrategyDecision, default_size: float) -> float:
    raw_size = default_size if decision.size is None else decision.size
    try:
        order_size = float(raw_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"decision size must be numeric; received {raw_size!r}") from exc

    if not math.isfinite(order_size) or order_size <= 0.0:
        raise ValueError(f"decision size must be > 0 and finite; received {raw_size!r}")
    return round(order_size, 2)


def _build_position_summary(wallet: AccountInventory, epic: str) -> dict[str, Any]:
    position = wallet.position_for(epic)
    if position is None:
        return {
            "summary": "position=NO_POSITION",
            "side": None,
            "size": None,
            "entry_price": None,
            "count": 0,
        }

    return {
        "summary": wallet.position_summary(),
        "side": position.side,
        "size": position.size,
        "entry_price": position.entry_price,
        "count": len(wallet.positions),
    }


def _build_market_payload(
    *,
    epic: str,
    market_spec: Any,
    tick: Any,
    bar_builder: LiveBarBuilder,
) -> dict[str, Any]:
    current_bar = bar_builder.current_bar
    return {
        "epic": epic,
        "name": market_spec.name,
        "contract_size": market_spec.contract_size,
        "margin_factor": market_spec.margin_factor,
        "bid": tick.bid,
        "ask": tick.ask,
        "mid": tick.mid,
        "spread": tick.spread,
        "volume": None if current_bar is None else current_bar.volume,
        "bar": {
            "open": None if current_bar is None else current_bar.open,
            "high": None if current_bar is None else current_bar.high,
            "low": None if current_bar is None else current_bar.low,
            "close": None if current_bar is None else current_bar.close,
            "volume": None if current_bar is None else current_bar.volume,
        },
    }


def _bars_frame_with_live_bar(bar_builder: LiveBarBuilder) -> pd.DataFrame:
    bars = bar_builder.bars_frame()
    current_bar = bar_builder.current_bar
    if current_bar is None:
        return bars

    live_row = pd.DataFrame(
        [
            {
                "time": current_bar.minute,
                "open": current_bar.open,
                "high": current_bar.high,
                "low": current_bar.low,
                "close": current_bar.close,
                "volume": float("nan") if current_bar.volume is None else current_bar.volume,
            }
        ]
    )
    if bars.empty:
        return live_row
    return pd.concat([bars, live_row], ignore_index=True)


def _build_strategy_context(
    bar_builder: LiveBarBuilder,
    *,
    include_current_bar: bool,
    current_bar_time: pd.Timestamp | None = None,
) -> StrategyContext:
    bars = _bars_frame_with_live_bar(bar_builder) if include_current_bar else bar_builder.bars_frame()
    if bars.empty:
        closes = pd.Series(dtype="float64")
    else:
        closes = pd.Series(bars["close"].astype("float64").to_numpy(), dtype="float64")

    if current_bar_time is None:
        current_bar = bar_builder.current_bar
        current_bar_time = None if current_bar is None else current_bar.minute

    return StrategyContext(bars=bars, closes=closes, current_bar_time=current_bar_time)


def _build_account_payload(wallet: AccountInventory, current_price: float) -> dict[str, Any]:
    snapshot = wallet.snapshot(current_price)
    snapshot["position_count"] = len(wallet.positions)
    return snapshot


def _build_execution_payload(
    *,
    workflow_config: WorkflowConfig,
    engine: SimulatedExecutionEngine,
    market_state: MarketTradingState,
) -> dict[str, Any]:
    return {
        "stream": "on" if workflow_config.stream_price_enabled else "off",
        "strategy": "on" if workflow_config.strategy_process_enabled else "off",
        "execution": "on" if workflow_config.execution_process_enabled else "off",
        "strategy_enabled": workflow_config.strategy_process_enabled,
        "execution_enabled": workflow_config.execution_process_enabled,
        "pending_orders": engine.pending_count,
        "market_status": market_state.market_status,
        "market_open": market_state.is_open,
        "market": "open" if market_state.is_open else "closed",
    }


def _build_live_snapshot(
    *,
    tick: Any,
    workflow_config: WorkflowConfig,
    market_spec: Any,
    bar_builder: LiveBarBuilder,
    wallet: AccountInventory,
    engine: SimulatedExecutionEngine,
    strategy_snapshot: StrategyLiveSnapshot,
    session_tracker: RangeTracker,
    trade_tracker: RangeTracker,
    market_state: MarketTradingState,
) -> dict[str, Any]:
    market_payload = _build_market_payload(epic=workflow_config.epic, market_spec=market_spec, tick=tick, bar_builder=bar_builder)
    account_payload = _build_account_payload(wallet, tick.mid)
    execution_payload = _build_execution_payload(workflow_config=workflow_config, engine=engine, market_state=market_state)
    position_payload = _build_position_summary(wallet, workflow_config.epic)
    extrema_payload = {
        "session_high_equity": session_tracker.high_equity,
        "session_low_equity": session_tracker.low_equity,
        "session_high_price": session_tracker.high_price,
        "session_low_price": session_tracker.low_price,
        **trade_tracker.snapshot(),
    }

    return {
        "timestamp": tick.timestamp,
        "market": market_payload,
        "account": account_payload,
        "execution": execution_payload,
        "position": position_payload,
        "extrema": extrema_payload,
        "strategy": asdict(strategy_snapshot),
    }


def _load_workflow_config(config_path: Path) -> WorkflowConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Workflow config file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle) or {}

        if not isinstance(raw, dict):
            raise ValueError("root config must be a JSON object")

        workflow_data = raw.get("workflow", {})
        process_data = raw.get("processes", {})
        stream_data = raw.get("stream", {})
        terminal_data = raw.get("terminal", {})
        logging_data = raw.get("logging", {})

        if workflow_data is None:
            workflow_data = {}
        if process_data is None:
            process_data = {}
        if stream_data is None:
            stream_data = {}
        if terminal_data is None:
            terminal_data = {}
        if logging_data is None:
            logging_data = {}

        for section_name, section_value in (
            ("workflow", workflow_data),
            ("processes", process_data),
            ("stream", stream_data),
            ("terminal", terminal_data),
            ("logging", logging_data),
        ):
            if not isinstance(section_value, dict):
                raise ValueError(f"{section_name} must be a JSON object")

        strategy_overrides = _coerce_strategy_overrides(workflow_data.get("strategy_overrides"))

        stream_price_enabled = _parse_bool(process_data.get("stream_price", True), key="processes.stream_price")
        strategy_process_enabled = _parse_bool(process_data.get("strategy", False), key="processes.strategy")
        execution_process_enabled = _parse_bool(process_data.get("execution", True), key="processes.execution")

        epic_value = workflow_data.get("epic", DEFAULT_EPIC)
        if not isinstance(epic_value, str) or epic_value.strip() == "":
            raise ValueError("workflow.epic must be a non-empty string")

        strategy_name_value = workflow_data.get("strategy_name", "std_levels_touch_density")
        if strategy_name_value is not None and not isinstance(strategy_name_value, str):
            raise ValueError("workflow.strategy_name must be a string or null")

        return WorkflowConfig(
            config_path=config_path,
            enabled=_parse_bool(raw.get("enabled", True), key="enabled"),
            reload_interval_seconds=_parse_float(
                raw.get("reload_interval_seconds", 30.0),
                key="reload_interval_seconds",
                minimum=0.1,
            ),
            epic=epic_value,
            strategy_name=strategy_name_value,
            strategy_overrides=strategy_overrides,
            stream_price_enabled=stream_price_enabled,
            strategy_process_enabled=strategy_process_enabled,
            execution_process_enabled=execution_process_enabled,
            stream_print_seconds=_parse_float(
                workflow_data.get("stream_print_seconds", DEFAULT_STREAM_PRINT_SECONDS),
                key="workflow.stream_print_seconds",
                minimum=0.1,
            ),
            stream=LivePriceFeedConfig(
                reconnect_delay_seconds=_parse_float(
                    stream_data.get("reconnect_delay_seconds", 1.0),
                    key="stream.reconnect_delay_seconds",
                    minimum=0.0,
                ),
                max_consecutive_failures=_parse_int(
                    stream_data.get("max_consecutive_failures", 5),
                    key="stream.max_consecutive_failures",
                    minimum=1,
                ),
                poll_interval_seconds=_parse_float(
                    stream_data.get("poll_interval_seconds", 0.2),
                    key="stream.poll_interval_seconds",
                    minimum=0.01,
                ),
                ticker_timeout_seconds=_parse_float(
                    stream_data.get("ticker_timeout_seconds", 10.0),
                    key="stream.ticker_timeout_seconds",
                    minimum=1.0,
                ),
                buffer_size=_parse_int(
                    stream_data.get("buffer_size", 1024),
                    key="stream.buffer_size",
                    minimum=1,
                ),
            ),
            terminal_display=TerminalDisplayConfig(
                show_timestamp=_parse_bool(terminal_data.get("show_timestamp", True), key="terminal.show_timestamp"),
                show_price=_parse_bool(terminal_data.get("show_price", True), key="terminal.show_price"),
                show_spread=_parse_bool(terminal_data.get("show_spread", True), key="terminal.show_spread"),
                show_market=_parse_bool(terminal_data.get("show_market", True), key="terminal.show_market"),
                show_stream_status=_parse_bool(
                    terminal_data.get("show_stream_status", True),
                    key="terminal.show_stream_status",
                ),
                show_strategy_status=_parse_bool(
                    terminal_data.get("show_strategy_status", True),
                    key="terminal.show_strategy_status",
                ),
                show_execution_status=_parse_bool(
                    terminal_data.get("show_execution_status", True),
                    key="terminal.show_execution_status",
                ),
                show_positions=_parse_bool(terminal_data.get("show_positions", True), key="terminal.show_positions"),
                show_unrealized_pnl=_parse_bool(
                    terminal_data.get("show_unrealized_pnl", True),
                    key="terminal.show_unrealized_pnl",
                ),
                show_equity=_parse_bool(terminal_data.get("show_equity", True), key="terminal.show_equity"),
                show_signal_criteria=_parse_bool(
                    terminal_data.get("show_signal_criteria", True),
                    key="terminal.show_signal_criteria",
                ),
                show_market_metadata=_parse_bool(
                    terminal_data.get("show_market_metadata", True),
                    key="terminal.show_market_metadata",
                ),
                show_account_metadata=_parse_bool(
                    terminal_data.get("show_account_metadata", True),
                    key="terminal.show_account_metadata",
                ),
                show_trade_extremes=_parse_bool(
                    terminal_data.get("show_trade_extremes", True),
                    key="terminal.show_trade_extremes",
                ),
                show_volume=_parse_bool(terminal_data.get("show_volume", True), key="terminal.show_volume"),
                show_bar_ohlc=_parse_bool(terminal_data.get("show_bar_ohlc", True), key="terminal.show_bar_ohlc"),
            ),
            logging=WorkflowLoggingConfig(
                csv_path=_resolve_path(config_path.parent, logging_data.get("csv_path"), DEFAULT_LOG_PATH),
            ),
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in config file {config_path}: {exc}") from exc
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid workflow config at {config_path}: {exc}") from exc


def _strategy_signature(config: WorkflowConfig) -> tuple[Any, ...]:
    overrides = config.strategy_overrides or {}
    return (
        config.strategy_process_enabled,
        config.strategy_name,
        tuple(sorted(overrides.items())),
    )


def run_live_virtual_trading(config: WorkflowConfig | None = None) -> tuple[AccountInventory, Any]:
    workflow_config = config or _load_workflow_config(DEFAULT_CONFIG_PATH)
    service = build_service(load_credentials())
    market_spec = load_market_spec(service, workflow_config.epic)
    market_state = market_trading_state(get_market_by_epic(service, workflow_config.epic))

    wallet = AccountInventory()
    feed = LivePriceFeed(service, workflow_config.epic, workflow_config.stream)
    bar_builder = LiveBarBuilder()
    engine = SimulatedExecutionEngine()
    logger = CsvRunLogger(workflow_config.logging.csv_path)
    terminal = RichLiveTerminal(workflow_config.terminal_display, title=f"{market_spec.name} | {workflow_config.strategy_name or 'disabled'}")

    historical_bars = load_or_fetch_historical_ohlcv(service, workflow_config.epic)
    bar_builder.load_history(historical_bars)

    strategy = (
        DEFAULT_STRATEGY_REGISTRY.create(workflow_config.strategy_name, workflow_config.strategy_overrides)
        if workflow_config.strategy_process_enabled
        else DEFAULT_STRATEGY_REGISTRY.create("disabled")
    )

    strategy_signature = _strategy_signature(workflow_config)
    strategy_snapshot = strategy.live_snapshot(_build_strategy_context(bar_builder, include_current_bar=True), None)
    last_decision: StrategyDecision | None = None

    session_tracker = RangeTracker()
    trade_tracker = RangeTracker()
    config_reload_at = time.monotonic() + workflow_config.reload_interval_seconds
    market_closed_retry_seconds = max(DEFAULT_MARKET_CLOSED_SLEEP_SECONDS, workflow_config.stream.poll_interval_seconds)
    last_market_status: str | None = None

    if workflow_config.stream_price_enabled and market_state.is_open:
        feed.start()

    last_print = 0.0
    try:
        with terminal:
            while True:
                now_mono = time.monotonic()
                if now_mono >= config_reload_at:
                    refreshed_config = _load_workflow_config(workflow_config.config_path)
                    if refreshed_config != workflow_config:
                        workflow_config = refreshed_config
                        feed.config = workflow_config.stream
                        terminal.display = workflow_config.terminal_display

                        if workflow_config.logging.csv_path != logger.csv_path:
                            logger = CsvRunLogger(workflow_config.logging.csv_path)

                        new_signature = _strategy_signature(workflow_config)
                        if new_signature != strategy_signature:
                            strategy = (
                                DEFAULT_STRATEGY_REGISTRY.create(workflow_config.strategy_name, workflow_config.strategy_overrides)
                                if workflow_config.strategy_process_enabled
                                else DEFAULT_STRATEGY_REGISTRY.create("disabled")
                            )
                            strategy_signature = new_signature
                            strategy_snapshot = strategy.live_snapshot(_build_strategy_context(bar_builder, include_current_bar=True), last_decision)

                    try:
                        refreshed_market_state = market_trading_state(get_market_by_epic(service, workflow_config.epic))
                    except (IGException, OSError, ValueError, TypeError) as exc:
                        refreshed_market_state = market_state
                        refreshed_market_state = MarketTradingState(
                            market_status=market_state.market_status,
                            is_open=market_state.is_open,
                            reason=f"status unavailable: {exc}",
                        )

                    market_state = refreshed_market_state
                    if market_state.market_status != last_market_status:
                        if market_state.is_open:
                            terminal.message(f"Market open for {workflow_config.epic} ({market_state.market_status}); resuming live trading.", style="green")
                        else:
                            terminal.message(f"Market closed for {workflow_config.epic} ({market_state.market_status}); pausing until open.", style="yellow")
                        last_market_status = market_state.market_status

                    if market_state.is_open:
                        if workflow_config.stream_price_enabled:
                            if not feed.is_running():
                                feed.start()
                        elif feed.is_running():
                            feed.stop()
                    elif feed.is_running():
                        feed.stop()

                    if not workflow_config.enabled:
                        terminal.message("Workflow disabled by configuration; stopping.", style="yellow")
                        break

                    config_reload_at = now_mono + max(workflow_config.reload_interval_seconds, 1.0)

                if not workflow_config.enabled:
                    break

                if not market_state.is_open:
                    if feed.is_running():
                        feed.stop()
                    time.sleep(market_closed_retry_seconds)
                    continue

                if workflow_config.stream_price_enabled:
                    if not feed.is_running():
                        feed.start()
                elif feed.is_running():
                    feed.stop()

                tick = feed.read_tick()
                if tick is None:
                    time.sleep(workflow_config.stream.poll_interval_seconds)
                    continue

                pre_execution_snapshot = wallet.snapshot(tick.mid)
                if wallet.position_for(workflow_config.epic) is not None:
                    trade_tracker.update(timestamp=tick.timestamp, price=tick.mid, equity=pre_execution_snapshot["equity"])

                receipts: list[ExecutionReceipt] = []
                if workflow_config.execution_process_enabled:
                    try:
                        receipts = engine.process_pending(tick=tick, wallet=wallet, market_spec=market_spec)
                    except (RuntimeError, ValueError, OSError) as exc:
                        message = f"Execution error: {exc}"
                        terminal.message(message, style="red")
                        live_snapshot = _build_live_snapshot(
                            tick=tick,
                            workflow_config=workflow_config,
                            market_spec=market_spec,
                            bar_builder=bar_builder,
                            wallet=wallet,
                            engine=engine,
                            strategy_snapshot=strategy_snapshot,
                            session_tracker=session_tracker,
                            trade_tracker=trade_tracker,
                            market_state=market_state,
                        )
                        logger.log_snapshot(live_snapshot, event_type="error", note=message)
                        raise RuntimeError(message) from exc

                post_execution_snapshot = wallet.snapshot(tick.mid)
                for receipt in receipts:
                    if receipt.action == "open":
                        trade_tracker.start(
                            timestamp=receipt.filled_at,
                            price=receipt.fill_price,
                            equity=post_execution_snapshot["equity"],
                        )
                        open_snapshot = _build_live_snapshot(
                            tick=tick,
                            workflow_config=workflow_config,
                            market_spec=market_spec,
                            bar_builder=bar_builder,
                            wallet=wallet,
                            engine=engine,
                            strategy_snapshot=strategy_snapshot,
                            session_tracker=session_tracker,
                            trade_tracker=trade_tracker,
                            market_state=market_state,
                        )
                        logger.log_snapshot(
                            open_snapshot,
                            event_type="position_open",
                            note=f"filled {receipt.side} at {receipt.fill_price:.2f}",
                            extra={"receipt": asdict(receipt), "trade_extrema": trade_tracker.snapshot()},
                        )
                    elif receipt.action == "close":
                        close_snapshot = _build_live_snapshot(
                            tick=tick,
                            workflow_config=workflow_config,
                            market_spec=market_spec,
                            bar_builder=bar_builder,
                            wallet=wallet,
                            engine=engine,
                            strategy_snapshot=strategy_snapshot,
                            session_tracker=session_tracker,
                            trade_tracker=trade_tracker,
                            market_state=market_state,
                        )
                        logger.log_snapshot(
                            close_snapshot,
                            event_type="position_close",
                            note=f"filled {receipt.side} at {receipt.fill_price:.2f}",
                            extra={"receipt": asdict(receipt), "trade_extrema": trade_tracker.snapshot()},
                        )
                        trade_tracker.reset()

                finished_bar = bar_builder.update(tick.timestamp, tick.mid)
                if finished_bar is not None:
                    context = _build_strategy_context(bar_builder, include_current_bar=False, current_bar_time=finished_bar.minute)
                    try:
                        decision = strategy.on_bar(context)
                    except (RuntimeError, ValueError) as exc:
                        message = f"Strategy error for {strategy.name}: {exc}"
                        terminal.message(message, style="red")
                        logger.log_snapshot(
                            _build_live_snapshot(
                                tick=tick,
                                workflow_config=workflow_config,
                                market_spec=market_spec,
                                bar_builder=bar_builder,
                                wallet=wallet,
                                engine=engine,
                                strategy_snapshot=strategy_snapshot,
                                session_tracker=session_tracker,
                                trade_tracker=trade_tracker,
                                market_state=market_state,
                            ),
                            event_type="error",
                            note=message,
                        )
                        raise RuntimeError(message) from exc

                    strategy_snapshot = strategy.live_snapshot(context, decision)
                    if decision is not None:
                        last_decision = decision

                    if workflow_config.execution_process_enabled:
                        active_position = wallet.position_for(workflow_config.epic)
                        if (
                            decision is not None
                            and decision.action == "enter"
                            and decision.side is not None
                            and active_position is None
                            and engine.pending_count == 0
                        ):
                            try:
                                order_size = _resolve_order_size(decision, engine.settings.default_order_size)
                            except ValueError as exc:
                                message = f"Invalid order size from {strategy.name}: {exc}"
                                terminal.message(message, style="red")
                                logger.log_snapshot(
                                    _build_live_snapshot(
                                        tick=tick,
                                        workflow_config=workflow_config,
                                        market_spec=market_spec,
                                        bar_builder=bar_builder,
                                        wallet=wallet,
                                        engine=engine,
                                        strategy_snapshot=strategy_snapshot,
                                        session_tracker=session_tracker,
                                        trade_tracker=trade_tracker,
                                        market_state=market_state,
                                    ),
                                    event_type="error",
                                    note=message,
                                )
                                raise RuntimeError(message) from exc
                            try:
                                engine.submit_order(
                                    OrderRequest(
                                        action="open",
                                        side=decision.side,
                                        epic=workflow_config.epic,
                                        size=order_size,
                                        submitted_at=finished_bar.minute,
                                        reference_price=finished_bar.close,
                                    )
                                )
                            except (RuntimeError, ValueError) as exc:
                                message = f"Execution error: {exc}"
                                terminal.message(message, style="red")
                                logger.log_snapshot(
                                    _build_live_snapshot(
                                        tick=tick,
                                        workflow_config=workflow_config,
                                        market_spec=market_spec,
                                        bar_builder=bar_builder,
                                        wallet=wallet,
                                        engine=engine,
                                        strategy_snapshot=strategy_snapshot,
                                        session_tracker=session_tracker,
                                        trade_tracker=trade_tracker,
                                        market_state=market_state,
                                    ),
                                    event_type="error",
                                    note=message,
                                )
                                raise RuntimeError(message) from exc
                        elif (
                            decision is not None
                            and decision.action == "exit"
                            and active_position is not None
                            and engine.pending_count == 0
                        ):
                            exit_side: Side = "sell" if active_position.side == "buy" else "buy"
                            try:
                                engine.submit_order(
                                    OrderRequest(
                                        action="close",
                                        side=exit_side,
                                        epic=workflow_config.epic,
                                        size=active_position.size,
                                        submitted_at=finished_bar.minute,
                                        reference_price=finished_bar.close,
                                    )
                                )
                            except (RuntimeError, ValueError) as exc:
                                message = f"Execution error: {exc}"
                                terminal.message(message, style="red")
                                logger.log_snapshot(
                                    _build_live_snapshot(
                                        tick=tick,
                                        workflow_config=workflow_config,
                                        market_spec=market_spec,
                                        bar_builder=bar_builder,
                                        wallet=wallet,
                                        engine=engine,
                                        strategy_snapshot=strategy_snapshot,
                                        session_tracker=session_tracker,
                                        trade_tracker=trade_tracker,
                                        market_state=market_state,
                                    ),
                                    event_type="error",
                                    note=message,
                                )
                                raise RuntimeError(message) from exc
                        elif (
                            decision is not None
                            and decision.action == "enter"
                            and decision.side is not None
                            and active_position is not None
                            and engine.pending_count == 0
                        ):
                            should_exit = (active_position.side == "buy" and decision.side == "sell") or (
                                active_position.side == "sell" and decision.side == "buy"
                            )
                            if should_exit:
                                exit_side = "sell" if active_position.side == "buy" else "buy"
                                try:
                                    engine.submit_order(
                                        OrderRequest(
                                            action="close",
                                            side=exit_side,
                                            epic=workflow_config.epic,
                                            size=active_position.size,
                                            submitted_at=finished_bar.minute,
                                            reference_price=finished_bar.close,
                                        )
                                    )
                                except (RuntimeError, ValueError) as exc:
                                    message = f"Execution error: {exc}"
                                    terminal.message(message, style="red")
                                    logger.log_snapshot(
                                        _build_live_snapshot(
                                            tick=tick,
                                            workflow_config=workflow_config,
                                            market_spec=market_spec,
                                            bar_builder=bar_builder,
                                            wallet=wallet,
                                            engine=engine,
                                            strategy_snapshot=strategy_snapshot,
                                            session_tracker=session_tracker,
                                            trade_tracker=trade_tracker,
                                            market_state=market_state,
                                        ),
                                        event_type="error",
                                        note=message,
                                    )
                                    raise RuntimeError(message) from exc

                current_snapshot = wallet.snapshot(tick.mid)
                session_tracker.update(timestamp=tick.timestamp, price=tick.mid, equity=current_snapshot["equity"])
                if wallet.position_for(workflow_config.epic) is not None:
                    trade_tracker.update(timestamp=tick.timestamp, price=tick.mid, equity=current_snapshot["equity"])

                now = time.time()
                if now - last_print >= workflow_config.stream_print_seconds:
                    live_strategy_snapshot = strategy.live_snapshot(
                        _build_strategy_context(bar_builder, include_current_bar=True),
                        last_decision,
                    )
                    strategy_snapshot = live_strategy_snapshot
                    live_snapshot = _build_live_snapshot(
                        tick=tick,
                        workflow_config=workflow_config,
                        market_spec=market_spec,
                        bar_builder=bar_builder,
                        wallet=wallet,
                        engine=engine,
                        strategy_snapshot=live_strategy_snapshot,
                        session_tracker=session_tracker,
                        trade_tracker=trade_tracker,
                        market_state=market_state,
                    )
                    terminal.update(live_snapshot)
                    last_print = now

                time.sleep(workflow_config.stream.poll_interval_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        feed.stop()

    return wallet, market_spec


def run_live_virtual_trading_forever(config: WorkflowConfig | None = None) -> None:
    restart_backoff_seconds = DEFAULT_RESTART_BACKOFF_SECONDS
    while True:
        try:
            run_live_virtual_trading(config)
            return
        except KeyboardInterrupt:
            raise
        except (IGException, RuntimeError, ValueError, OSError) as exc:
            print(f"Trading workflow stopped unexpectedly: {exc}", file=sys.stderr)
            time.sleep(restart_backoff_seconds)


def main() -> None:
    try:
        run_live_virtual_trading_forever()
    except IGException as exc:
        raise SystemExit(f"IG session failed: {exc}") from exc
    except (RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc

    console = RichLiveTerminal().console
    console.print("Trading workflow exited.")


if __name__ == "__main__":
    main()
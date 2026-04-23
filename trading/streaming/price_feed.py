from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Thread, current_thread
import time

import pandas as pd
from trading_ig import IGStreamService
from trading_ig.streamer.manager import StreamingManager

from trading.ig_nq_data import build_service, load_credentials


@dataclass(frozen=True)
class PriceTick:
    timestamp: pd.Timestamp
    bid: float
    ask: float
    mid: float

    @property
    def spread(self) -> float:
        return max(self.ask - self.bid, 0.0)


@dataclass
class MinuteBar:
    minute: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None

    def update(self, price: float) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price


@dataclass(frozen=True)
class LivePriceFeedConfig:
    reconnect_delay_seconds: float = 1.0
    max_consecutive_failures: int = 5
    poll_interval_seconds: float = 0.2
    ticker_timeout_seconds: float = 10.0
    buffer_size: int = 1024


class LiveBarBuilder:
    def __init__(self) -> None:
        self.current_bar: MinuteBar | None = None
        self.completed_bars: list[MinuteBar] = []

    def load_history(self, bars: pd.DataFrame) -> None:
        if bars.empty:
            return

        required_columns = {"time", "open", "high", "low", "close"}
        missing = required_columns.difference(bars.columns)
        if missing:
            raise ValueError(f"bars are missing required columns: {sorted(missing)}")

        ordered = bars.sort_values("time").reset_index(drop=True)
        for record in ordered.to_dict(orient="records"):
            volume_value = record.get("volume")
            self.completed_bars.append(
                MinuteBar(
                    minute=pd.Timestamp(record["time"]),
                    open=float(record["open"]),
                    high=float(record["high"]),
                    low=float(record["low"]),
                    close=float(record["close"]),
                    volume=float(volume_value) if volume_value is not None and pd.notna(volume_value) else None,
                )
            )

    def update(self, timestamp: pd.Timestamp, price: float) -> MinuteBar | None:
        minute = timestamp.floor("min")
        if self.current_bar is None:
            self.current_bar = MinuteBar(minute=minute, open=price, high=price, low=price, close=price)
            return None

        if minute == self.current_bar.minute:
            self.current_bar.update(price)
            return None

        finished_bar = self.current_bar
        self.completed_bars.append(finished_bar)
        self.current_bar = MinuteBar(minute=minute, open=price, high=price, low=price, close=price)
        return finished_bar

    def closes_series(self) -> pd.Series:
        if not self.completed_bars:
            return pd.Series(dtype="float64")
        idx = [bar.minute for bar in self.completed_bars]
        values = [bar.close for bar in self.completed_bars]
        return pd.Series(values, index=pd.DatetimeIndex(idx), dtype="float64")

    def bars_frame(self) -> pd.DataFrame:
        if not self.completed_bars:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        return pd.DataFrame(
            {
                "time": [bar.minute for bar in self.completed_bars],
                "open": [bar.open for bar in self.completed_bars],
                "high": [bar.high for bar in self.completed_bars],
                "low": [bar.low for bar in self.completed_bars],
                "close": [bar.close for bar in self.completed_bars],
                "volume": [bar.volume if bar.volume is not None else float("nan") for bar in self.completed_bars],
            }
        )


def _read_stream_tick(stream_manager: StreamingManager, epic: str) -> PriceTick:
    ticker = stream_manager.tickers.get(epic)
    if ticker is None:
        raise RuntimeError(f"No ticker object found for {epic}")

    bid = float(ticker.bid)
    ask = float(ticker.offer)
    timestamp = ticker.timestamp
    if pd.isna(bid) or pd.isna(ask) or timestamp is None:
        raise RuntimeError(f"Ticker fields are not ready for {epic}")

    mid = (bid + ask) / 2.0
    return PriceTick(timestamp=pd.Timestamp(timestamp), bid=bid, ask=ask, mid=mid)


class LivePriceFeed:
    def __init__(
        self,
        service: object,
        epic: str,
        config: LivePriceFeedConfig | None = None,
    ) -> None:
        self.service = service
        self.epic = epic
        self.config = config or LivePriceFeedConfig()
        self._stream_service: IGStreamService | None = None
        self._stream_manager: StreamingManager | None = None
        self._tick_queue: Queue[PriceTick] = Queue(maxsize=self.config.buffer_size)
        self._thread: Thread | None = None
        self._stop_event = Event()
        self._latest_tick: PriceTick | None = None
        self._last_tick_signature: tuple[pd.Timestamp, float, float] | None = None
        self._failure_count = 0
        self._fatal_error: RuntimeError | None = None

    @classmethod
    def from_credentials(
        cls,
        epic: str,
        config: LivePriceFeedConfig | None = None,
    ) -> "LivePriceFeed":
        service = build_service(load_credentials())
        return cls(service, epic, config=config)

    @property
    def latest_tick(self) -> PriceTick | None:
        return self._latest_tick

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    def start(self) -> None:
        if self.is_running():
            return

        self._stop_event.clear()
        self.connect()
        self._thread = Thread(target=self._run, name=f"price-feed-{self.epic}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not current_thread():
            thread.join(timeout=5.0)
        self.close()
        self._thread = None

    def connect(self) -> None:
        if self._stream_manager is not None:
            return

        session_data = self.service.read_session()
        if not isinstance(session_data, dict):
            raise RuntimeError("IG session response must be a mapping")
        account_id = session_data.get("accountId")
        if account_id is None or str(account_id).strip() == "":
            raise RuntimeError("IG session is missing required field accountId")

        stream_service = IGStreamService(self.service)
        stream_service.acc_number = str(account_id)
        stream_service.create_session()
        stream_manager = StreamingManager(stream_service)
        stream_manager.start_tick_subscription(self.epic)
        stream_manager.ticker(self.epic, timeout_length=int(max(self.config.ticker_timeout_seconds, 1.0)))

        self._stream_service = stream_service
        self._stream_manager = stream_manager
        self._failure_count = 0
        self._fatal_error = None

    def close(self) -> None:
        if self._stream_manager is None:
            return

        try:
            self._stream_manager.stop_subscriptions()
        finally:
            self._stream_manager = None
            self._stream_service = None
            self._failure_count = 0
            self._last_tick_signature = None

    def _restart(self) -> None:
        self.close()
        if self._stop_event.is_set():
            return
        time.sleep(self.config.reconnect_delay_seconds)
        self.connect()

    def _enqueue_tick(self, tick: PriceTick) -> None:
        try:
            self._tick_queue.put_nowait(tick)
        except Full:
            try:
                self._tick_queue.get_nowait()
            except Empty:
                pass
            self._tick_queue.put_nowait(tick)

    def _should_emit_tick(self, tick: PriceTick) -> bool:
        signature = (tick.timestamp, tick.bid, tick.ask)
        if signature == self._last_tick_signature:
            return False
        self._last_tick_signature = signature
        return True

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    if self._stream_manager is None:
                        self.connect()

                    assert self._stream_manager is not None
                    tick = _read_stream_tick(self._stream_manager, self.epic)
                    self._latest_tick = tick
                    self._failure_count = 0
                    if self._should_emit_tick(tick):
                        self._enqueue_tick(tick)
                    time.sleep(self.config.poll_interval_seconds)
                except KeyboardInterrupt:
                    self._stop_event.set()
                    raise
                except (ConnectionError, RuntimeError, ValueError):
                    self._failure_count += 1
                    if self._stop_event.is_set():
                        break
                    if self._failure_count >= self.config.max_consecutive_failures:
                        try:
                            self._restart()
                        except (ConnectionError, RuntimeError, ValueError) as restart_exc:
                            self._fatal_error = RuntimeError(
                                f"Price feed failed to restart for {self.epic}: {restart_exc}"
                            )
                            self._stop_event.set()
                            break
                    else:
                        time.sleep(self.config.poll_interval_seconds)
                except (AssertionError, AttributeError, OSError, TypeError) as exc:
                    self._fatal_error = RuntimeError(
                        f"Unexpected price feed failure for {self.epic}: {exc}"
                    )
                    self._stop_event.set()
                    break
        finally:
            self.close()

    def read_tick(self) -> PriceTick | None:
        if self._fatal_error is not None:
            fatal_error = self._fatal_error
            self._fatal_error = None
            raise fatal_error

        try:
            return self._tick_queue.get_nowait()
        except Empty:
            return None

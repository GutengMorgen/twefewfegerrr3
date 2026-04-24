from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading.strategies.base import (
    StrategyConfig,
    StrategyContext,
    StrategyDecision,
    StrategyLiveSnapshot,
    StrategyPlugin,
    StrategySignalCriterion,
)


@dataclass(frozen=True)
class Breakout0600Config(StrategyConfig):
    range_start_hour: int = 9
    range_start_minute: int = 30
    range_end_hour: int = 10
    range_end_minute: int = 0
    entry_hour: int = 10
    entry_minute: int = 1
    exit_hour: int = 15
    exit_minute: int = 59
    use_ema_filter: bool = True
    ema_fast_period: int = 9
    ema_slow_period: int = 21
    ema_trend_period: int = 100
    power_lookback: int = 20
    power_range_multiplier: float = 1.5
    power_body_ratio_min: float = 0.60
    power_close_edge_max: float = 0.20
    power_signal_max_bars: int = 3
    measured_move_multiplier: float = 1.0
    tick_size: float = 0.25
    tp_ticks: int = 0
    sl_ticks: int = 0


def _is_power_bar(
    is_long: bool,
    bar_open: float,
    bar_high: float,
    bar_low: float,
    bar_close: float,
    avg_range: float,
    cfg: Breakout0600Config,
) -> bool:
    bar_range = bar_high - bar_low
    if bar_range <= 0.0 or avg_range <= 0.0:
        return False

    if bar_range < avg_range * cfg.power_range_multiplier:
        return False

    body_ratio = abs(bar_close - bar_open) / bar_range
    if body_ratio < cfg.power_body_ratio_min:
        return False

    if is_long:
        close_to_high = (bar_high - bar_close) / bar_range
        return bar_close > bar_open and close_to_high <= cfg.power_close_edge_max

    close_to_low = (bar_close - bar_low) / bar_range
    return bar_close < bar_open and close_to_low <= cfg.power_close_edge_max


def _minute_of_day_array(time_values: pd.DatetimeIndex) -> np.ndarray:
    return (time_values.hour * 60 + time_values.minute).astype(np.int32)


def _day_id_array(time_values: pd.DatetimeIndex) -> np.ndarray:
    return time_values.normalize().to_numpy(dtype="datetime64[D]").astype(np.int64)


def _ema_array(price: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (float(span) + 1.0)
    out = np.empty(price.size, dtype=np.float64)
    out[0] = float(price[0])
    for index in range(1, price.size):
        out[index] = alpha * float(price[index]) + (1.0 - alpha) * out[index - 1]
    return out


class Breakout0600Strategy(StrategyPlugin):
    name = "breakout_0600_0730"

    def __init__(self, config: Breakout0600Config):
        super().__init__(config)
        self._validate_config()
        self._reset_state()
        self._last_bar_count = 0

    def _validate_config(self) -> None:
        config: Breakout0600Config = self.config
        if config.range_start_hour < 0 or config.range_start_hour > 23:
            raise ValueError("range_start_hour must be between 0 and 23")
        if config.range_end_hour < 0 or config.range_end_hour > 23:
            raise ValueError("range_end_hour must be between 0 and 23")
        if config.entry_hour < 0 or config.entry_hour > 23:
            raise ValueError("entry_hour must be between 0 and 23")
        if config.exit_hour < 0 or config.exit_hour > 23:
            raise ValueError("exit_hour must be between 0 and 23")
        if config.power_lookback < 1:
            raise ValueError("power_lookback must be >= 1")
        if config.ema_fast_period < 1 or config.ema_slow_period < 1 or config.ema_trend_period < 1:
            raise ValueError("EMA periods must be >= 1")
        if config.power_signal_max_bars < 1:
            raise ValueError("power_signal_max_bars must be >= 1")
        if config.tick_size <= 0.0:
            raise ValueError("tick_size must be > 0")
        if config.power_range_multiplier <= 0.0:
            raise ValueError("power_range_multiplier must be > 0")
        if not 0.0 <= config.power_body_ratio_min <= 1.0:
            raise ValueError("power_body_ratio_min must be between 0 and 1")
        if not 0.0 <= config.power_close_edge_max <= 1.0:
            raise ValueError("power_close_edge_max must be between 0 and 1")
        if config.measured_move_multiplier <= 0.0:
            raise ValueError("measured_move_multiplier must be > 0")
        if config.tp_ticks < 0 or config.sl_ticks < 0:
            raise ValueError("tp_ticks and sl_ticks must be >= 0")

    def _reset_state(self) -> None:
        self._current_day_id: int | None = None
        self._range_high = 0.0
        self._range_low = 0.0
        self._range_initialized = False
        self._signal_locked = False
        self._range_window = np.zeros(int(self.config.power_lookback), dtype=np.float64)
        self._range_window_count = 0
        self._range_window_pos = 0
        self._range_window_sum = 0.0
        self._post_bar_index = 0
        self._last_power_long_index = -100000
        self._last_power_short_index = -100000
        self._last_power_long_high = 0.0
        self._last_power_short_low = 0.0
        self._in_position = False
        self._pos_side = 0
        self._stop_price = 0.0
        self._target_price = 0.0

    @property
    def range_start(self) -> int:
        config: Breakout0600Config = self.config
        return config.range_start_hour * 60 + config.range_start_minute

    @property
    def range_end(self) -> int:
        config: Breakout0600Config = self.config
        return config.range_end_hour * 60 + config.range_end_minute

    @property
    def entry_start(self) -> int:
        config: Breakout0600Config = self.config
        return config.entry_hour * 60 + config.entry_minute

    @property
    def session_exit(self) -> int:
        config: Breakout0600Config = self.config
        return config.exit_hour * 60 + config.exit_minute

    def _extract_bars(self, context: StrategyContext) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        bars = context.bars
        required_columns = {"time", "open", "high", "low", "close"}
        if bars.empty or not required_columns.issubset(bars.columns):
            return None

        time_index = pd.DatetimeIndex(pd.to_datetime(bars["time"]))
        if time_index.tz is not None:
            time_index = time_index.tz_convert(None)

        open_values = bars["open"].to_numpy(dtype=np.float64, copy=False)
        high_values = bars["high"].to_numpy(dtype=np.float64, copy=False)
        low_values = bars["low"].to_numpy(dtype=np.float64, copy=False)
        close_values = bars["close"].to_numpy(dtype=np.float64, copy=False)
        return time_index, open_values, high_values, low_values, close_values

    def _ema_ok(self, is_long: bool, close_price: float, ema_fast: float, ema_slow: float, ema_trend: float) -> bool:
        config: Breakout0600Config = self.config
        if not config.use_ema_filter:
            return True

        if is_long:
            return ema_fast >= ema_slow and ema_slow >= ema_trend and close_price >= ema_fast
        return ema_fast <= ema_slow and ema_slow <= ema_trend and close_price <= ema_fast

    def _clear_position_state(self) -> None:
        self._in_position = False
        self._pos_side = 0
        self._stop_price = 0.0
        self._target_price = 0.0

    def _advance_bar(
        self,
        *,
        index: int,
        minute_of_day: np.ndarray,
        day_id: np.ndarray,
        open_values: np.ndarray,
        high_values: np.ndarray,
        low_values: np.ndarray,
        close_values: np.ndarray,
        ema_fast: np.ndarray,
        ema_slow: np.ndarray,
        ema_trend: np.ndarray,
    ) -> StrategyDecision | None:
        config: Breakout0600Config = self.config
        minute = int(minute_of_day[index])
        current_day = int(day_id[index])
        decision: StrategyDecision | None = None

        if self._current_day_id is None:
            self._current_day_id = current_day
        elif current_day != self._current_day_id:
            if self._in_position:
                decision = StrategyDecision(
                    action="exit",
                    side="sell" if self._pos_side == 1 else "buy",
                    reason="new_day_exit_long" if self._pos_side == 1 else "new_day_exit_short",
                )
                self._clear_position_state()

            self._current_day_id = current_day
            self._range_high = 0.0
            self._range_low = 0.0
            self._range_initialized = False
            self._signal_locked = False
            self._range_window.fill(0.0)
            self._range_window_count = 0
            self._range_window_pos = 0
            self._range_window_sum = 0.0
            self._post_bar_index = 0
            self._last_power_long_index = -100000
            self._last_power_short_index = -100000
            self._last_power_long_high = 0.0
            self._last_power_short_low = 0.0

        if self._in_position:
            if self._pos_side == 1:
                stop_hit = low_values[index] <= self._stop_price
                target_hit = high_values[index] >= self._target_price
                if stop_hit or target_hit or minute >= self.session_exit:
                    decision = StrategyDecision(
                        action="exit",
                        side="sell",
                        reason="stop_hit" if stop_hit else "target_hit" if target_hit else "session_exit_long",
                    )
                    self._clear_position_state()
            else:
                stop_hit = high_values[index] >= self._stop_price
                target_hit = low_values[index] <= self._target_price
                if stop_hit or target_hit or minute >= self.session_exit:
                    decision = StrategyDecision(
                        action="exit",
                        side="buy",
                        reason="stop_hit" if stop_hit else "target_hit" if target_hit else "session_exit_short",
                    )
                    self._clear_position_state()

        if self.range_start <= minute <= self.range_end:
            if not self._range_initialized:
                self._range_high = float(high_values[index])
                self._range_low = float(low_values[index])
                self._range_initialized = True
            else:
                self._range_high = max(self._range_high, float(high_values[index]))
                self._range_low = min(self._range_low, float(low_values[index]))
            return decision

        if not self._range_initialized:
            return decision

        if minute < self.entry_start or minute > self.session_exit:
            return decision

        if self._signal_locked:
            return decision

        current_range = float(high_values[index] - low_values[index])
        avg_range = self._range_window_sum / float(self._range_window_count) if self._range_window_count > 0 else 0.0

        bar_open = float(open_values[index])
        bar_high = float(high_values[index])
        bar_low = float(low_values[index])
        bar_close = float(close_values[index])

        broke_up = bar_high > self._range_high
        broke_down = bar_low < self._range_low

        power_long = False
        power_short = False
        if current_range > 0.0 and avg_range > 0.0 and current_range >= avg_range * config.power_range_multiplier:
            power_long = _is_power_bar(True, bar_open, bar_high, bar_low, bar_close, avg_range, config)
            power_short = _is_power_bar(False, bar_open, bar_high, bar_low, bar_close, avg_range, config)

        has_signal = False
        is_long = False
        signal_price = 0.0

        if broke_up and power_long and self._ema_ok(True, bar_close, ema_fast[index], ema_slow[index], ema_trend[index]):
            has_signal = True
            is_long = True
            signal_price = max(self._range_high, bar_open)
        elif broke_down and power_short and self._ema_ok(False, bar_close, ema_fast[index], ema_slow[index], ema_trend[index]):
            has_signal = True
            is_long = False
            signal_price = min(self._range_low, bar_open)
        elif (
            self._last_power_long_index > -100000
            and (self._post_bar_index - self._last_power_long_index) <= config.power_signal_max_bars
            and broke_up
            and bar_high > self._last_power_long_high
            and self._ema_ok(True, bar_close, ema_fast[index], ema_slow[index], ema_trend[index])
        ):
            has_signal = True
            is_long = True
            signal_price = max(max(self._range_high, self._last_power_long_high), bar_open)
        elif (
            self._last_power_short_index > -100000
            and (self._post_bar_index - self._last_power_short_index) <= config.power_signal_max_bars
            and broke_down
            and bar_low < self._last_power_short_low
            and self._ema_ok(False, bar_close, ema_fast[index], ema_slow[index], ema_trend[index])
        ):
            has_signal = True
            is_long = False
            signal_price = min(min(self._range_low, self._last_power_short_low), bar_open)

        if has_signal:
            self._signal_locked = True
            tick = max(config.tick_size, np.finfo(np.float64).eps)
            range_width = max((self._range_high - self._range_low) * config.measured_move_multiplier, tick)
            entry_price = bar_close

            if is_long:
                stop_price = min(
                    self._range_low + config.sl_ticks * tick,
                    signal_price - tick,
                    entry_price - tick,
                )
                target_price = max(
                    signal_price + range_width + config.tp_ticks * tick,
                    signal_price + tick,
                    entry_price + tick,
                )
                if stop_price < entry_price < target_price:
                    self._in_position = True
                    self._pos_side = 1
                    self._stop_price = stop_price
                    self._target_price = target_price
                    return StrategyDecision(action="enter", side="buy", reason="breakout_long_entry")
            else:
                stop_price = max(
                    self._range_high - config.sl_ticks * tick,
                    signal_price + tick,
                    entry_price + tick,
                )
                target_price = min(
                    signal_price - range_width - config.tp_ticks * tick,
                    signal_price - tick,
                    entry_price - tick,
                )
                if target_price < entry_price < stop_price:
                    self._in_position = True
                    self._pos_side = -1
                    self._stop_price = stop_price
                    self._target_price = target_price
                    return StrategyDecision(action="enter", side="sell", reason="breakout_short_entry")
            return decision

        if power_long:
            self._last_power_long_index = self._post_bar_index
            self._last_power_long_high = bar_high

        if power_short:
            self._last_power_short_index = self._post_bar_index
            self._last_power_short_low = bar_low

        range_value = max(0.0, current_range)
        if self._range_window_count < int(self.config.power_lookback):
            self._range_window[self._range_window_pos] = range_value
            self._range_window_sum += range_value
            self._range_window_count += 1
        else:
            self._range_window_sum -= self._range_window[self._range_window_pos]
            self._range_window[self._range_window_pos] = range_value
            self._range_window_sum += range_value
        self._range_window_pos = (self._range_window_pos + 1) % int(self.config.power_lookback)
        self._post_bar_index += 1
        return decision

    def _sync_state(self, context: StrategyContext) -> StrategyDecision | None:
        extracted = self._extract_bars(context)
        if extracted is None:
            return StrategyDecision(action="hold", reason="warmup")

        time_index, open_values, high_values, low_values, close_values = extracted
        n = int(close_values.size)
        if n == 0:
            return StrategyDecision(action="hold", reason="warmup")

        if n < self._last_bar_count:
            self._reset_state()
            self._last_bar_count = 0

        minute_of_day = _minute_of_day_array(time_index)
        day_id = _day_id_array(time_index)
        ema_fast = _ema_array(close_values, self.config.ema_fast_period)
        ema_slow = _ema_array(close_values, self.config.ema_slow_period)
        ema_trend = _ema_array(close_values, self.config.ema_trend_period)

        last_decision: StrategyDecision | None = None
        for index in range(self._last_bar_count, n):
            decision = self._advance_bar(
                index=index,
                minute_of_day=minute_of_day,
                day_id=day_id,
                open_values=open_values,
                high_values=high_values,
                low_values=low_values,
                close_values=close_values,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                ema_trend=ema_trend,
            )
            if decision is not None:
                last_decision = decision

        self._last_bar_count = n
        if last_decision is not None:
            return last_decision
        return StrategyDecision(action="hold", reason="no_signal")

    def on_bar(self, context: StrategyContext) -> StrategyDecision:
        decision = self._sync_state(context)
        if decision is None:
            return StrategyDecision(action="hold", reason="no_signal")
        return decision

    def live_snapshot(self, context: StrategyContext, decision: StrategyDecision | None = None) -> StrategyLiveSnapshot:
        extracted = self._extract_bars(context)
        if extracted is None:
            return self.warmup_snapshot(
                decision=decision,
                metadata={
                    "range_start_hour": self.config.range_start_hour,
                    "range_start_minute": self.config.range_start_minute,
                    "range_end_hour": self.config.range_end_hour,
                    "range_end_minute": self.config.range_end_minute,
                    "entry_hour": self.config.entry_hour,
                    "entry_minute": self.config.entry_minute,
                    "exit_hour": self.config.exit_hour,
                    "exit_minute": self.config.exit_minute,
                    "use_ema_filter": self.config.use_ema_filter,
                    "ema_fast_period": self.config.ema_fast_period,
                    "ema_slow_period": self.config.ema_slow_period,
                    "ema_trend_period": self.config.ema_trend_period,
                    "power_lookback": self.config.power_lookback,
                    "power_range_multiplier": self.config.power_range_multiplier,
                    "power_body_ratio_min": self.config.power_body_ratio_min,
                    "power_close_edge_max": self.config.power_close_edge_max,
                    "power_signal_max_bars": self.config.power_signal_max_bars,
                    "measured_move_multiplier": self.config.measured_move_multiplier,
                    "tick_size": self.config.tick_size,
                    "tp_ticks": self.config.tp_ticks,
                    "sl_ticks": self.config.sl_ticks,
                },
            )

        time_index, open_values, high_values, low_values, close_values = extracted
        n = int(close_values.size)
        if n == 0:
            return self.warmup_snapshot(decision=decision, metadata={"power_lookback": self.config.power_lookback})

        minute_of_day = _minute_of_day_array(time_index)
        ema_fast = _ema_array(close_values, self.config.ema_fast_period)
        ema_slow = _ema_array(close_values, self.config.ema_slow_period)
        ema_trend = _ema_array(close_values, self.config.ema_trend_period)

        current_index = n - 1
        minute = int(minute_of_day[current_index])
        bar_open = float(open_values[current_index])
        bar_high = float(high_values[current_index])
        bar_low = float(low_values[current_index])
        bar_close = float(close_values[current_index])
        current_range = float(bar_high - bar_low)
        avg_range = self._range_window_sum / float(self._range_window_count) if self._range_window_count > 0 else 0.0

        broke_up = self._range_initialized and bar_high > self._range_high
        broke_down = self._range_initialized and bar_low < self._range_low
        power_long = _is_power_bar(True, bar_open, bar_high, bar_low, bar_close, avg_range, self.config)
        power_short = _is_power_bar(False, bar_open, bar_high, bar_low, bar_close, avg_range, self.config)
        long_ema_ok = self._ema_ok(True, bar_close, ema_fast[current_index], ema_slow[current_index], ema_trend[current_index])
        short_ema_ok = self._ema_ok(False, bar_close, ema_fast[current_index], ema_slow[current_index], ema_trend[current_index])

        criteria = (
            StrategySignalCriterion(
                kind="info",
                label="minute_of_day",
                current=minute,
                target=self.entry_start,
                comparator=">=",
                active=minute >= self.entry_start,
            ),
            StrategySignalCriterion(
                kind="info",
                label="range_initialized",
                current=int(self._range_initialized),
                target=1,
                comparator="=",
                active=self._range_initialized,
            ),
            StrategySignalCriterion(
                kind="entry",
                label="power_long",
                current=current_range,
                target=avg_range * self.config.power_range_multiplier if avg_range > 0.0 else None,
                comparator=">=",
                active=power_long,
            ),
            StrategySignalCriterion(
                kind="entry",
                label="power_short",
                current=current_range,
                target=avg_range * self.config.power_range_multiplier if avg_range > 0.0 else None,
                comparator=">=",
                active=power_short,
            ),
            StrategySignalCriterion(
                kind="entry",
                label="long_breakout",
                current=bar_high,
                target=self._range_high if self._range_initialized else None,
                comparator=">",
                active=broke_up and (power_long or self._last_power_long_index > -100000) and long_ema_ok,
            ),
            StrategySignalCriterion(
                kind="entry",
                label="short_breakout",
                current=bar_low,
                target=self._range_low if self._range_initialized else None,
                comparator="<",
                active=broke_down and (power_short or self._last_power_short_index > -100000) and short_ema_ok,
            ),
            StrategySignalCriterion(
                kind="exit",
                label="position",
                current=bar_close,
                target=self._stop_price if self._in_position else None,
                comparator="managed",
                active=self._in_position,
            ),
            StrategySignalCriterion(
                kind="info",
                label="signal_locked",
                current=int(self._signal_locked),
                target=0,
                comparator="=",
                active=self._signal_locked,
            ),
        )

        if self._in_position:
            state = "long" if self._pos_side == 1 else "short"
        else:
            state = "bullish" if long_ema_ok and not short_ema_ok else "bearish" if short_ema_ok and not long_ema_ok else "flat"

        return StrategyLiveSnapshot(
            name=self.name,
            state=state,
            decision=decision,
            criteria=criteria,
            metadata={
                "range_start_hour": self.config.range_start_hour,
                "range_start_minute": self.config.range_start_minute,
                "range_end_hour": self.config.range_end_hour,
                "range_end_minute": self.config.range_end_minute,
                "entry_hour": self.config.entry_hour,
                "entry_minute": self.config.entry_minute,
                "exit_hour": self.config.exit_hour,
                "exit_minute": self.config.exit_minute,
                "use_ema_filter": self.config.use_ema_filter,
                "ema_fast_period": self.config.ema_fast_period,
                "ema_slow_period": self.config.ema_slow_period,
                "ema_trend_period": self.config.ema_trend_period,
                "power_lookback": self.config.power_lookback,
                "power_range_multiplier": self.config.power_range_multiplier,
                "power_body_ratio_min": self.config.power_body_ratio_min,
                "power_close_edge_max": self.config.power_close_edge_max,
                "power_signal_max_bars": self.config.power_signal_max_bars,
                "measured_move_multiplier": self.config.measured_move_multiplier,
                "tick_size": self.config.tick_size,
                "tp_ticks": self.config.tp_ticks,
                "sl_ticks": self.config.sl_ticks,
                "current_day_id": None if self._current_day_id is None else int(self._current_day_id),
                "minute_of_day": minute,
                "range_high": self._range_high,
                "range_low": self._range_low,
                "range_initialized": self._range_initialized,
                "signal_locked": self._signal_locked,
                "in_position": self._in_position,
                "pos_side": self._pos_side,
                "stop_price": self._stop_price,
                "target_price": self._target_price,
                "avg_range": avg_range,
                "current_range": current_range,
                "last_power_long_index": self._last_power_long_index,
                "last_power_short_index": self._last_power_short_index,
                "last_power_long_high": self._last_power_long_high,
                "last_power_short_low": self._last_power_short_low,
                "bars_seen": n,
            },
            reason=decision.reason if decision is not None else None,
        )
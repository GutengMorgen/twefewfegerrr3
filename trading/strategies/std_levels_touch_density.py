from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading.helpers.touch_density import compute_touch_density_from_bars
from trading.strategies.base import (
    StrategyConfig,
    StrategyContext,
    StrategyDecision,
    StrategyLiveSnapshot,
    StrategyPlugin,
    StrategySignalCriterion,
)


@dataclass(frozen=True)
class StdLevelsTouchDensityConfig(StrategyConfig):
    lookback_days: int = 120
    kernel_width: int = 7
    entry_threshold: float = 0.25
    exit_threshold: float = 0.05
    max_hold_bars: int = 5
    signal_timeframe_minutes: int = 3


class StdLevelsTouchDensityStrategy(StrategyPlugin):
    name = "std_levels_touch_density"

    def __init__(self, config: StdLevelsTouchDensityConfig):
        super().__init__(config)
        self._validate_config()
        self._last_htf_count = 0
        self._in_position = False
        self._side = 0
        self._entry_index = 0

    def _validate_config(self) -> None:
        config: StdLevelsTouchDensityConfig = self.config
        if config.lookback_days < 1:
            raise ValueError("lookback_days must be >= 1")
        if config.kernel_width < 1:
            raise ValueError("kernel_width must be >= 1")
        if config.max_hold_bars < 1:
            raise ValueError("max_hold_bars must be >= 1")
        if config.entry_threshold <= 0.0:
            raise ValueError("entry_threshold must be > 0")
        if config.exit_threshold < 0.0:
            raise ValueError("exit_threshold must be >= 0")
        if config.exit_threshold >= config.entry_threshold:
            raise ValueError("exit_threshold must be smaller than entry_threshold")
        if config.signal_timeframe_minutes < 1:
            raise ValueError("signal_timeframe_minutes must be >= 1")

    def _signal_to_decision(self, score: float) -> StrategyDecision:
        config: StdLevelsTouchDensityConfig = self.config
        if self._in_position:
            held_bars = self._last_htf_count - self._entry_index
            if self._side == 1:
                if score <= config.exit_threshold or held_bars >= config.max_hold_bars:
                    self._in_position = False
                    self._side = 0
                    return StrategyDecision(action="exit", side="sell", reason="touch_density_exit_long")
            else:
                if score >= -config.exit_threshold or held_bars >= config.max_hold_bars:
                    self._in_position = False
                    self._side = 0
                    return StrategyDecision(action="exit", side="buy", reason="touch_density_exit_short")
            return StrategyDecision(action="hold", reason="touch_density_hold")

        if score >= config.entry_threshold:
            self._in_position = True
            self._side = 1
            self._entry_index = self._last_htf_count
            return StrategyDecision(action="enter", side="buy", reason="touch_density_long_entry")
        if score <= -config.entry_threshold:
            self._in_position = True
            self._side = -1
            self._entry_index = self._last_htf_count
            return StrategyDecision(action="enter", side="sell", reason="touch_density_short_entry")
        return StrategyDecision(action="hold", reason="touch_density_flat")

    def _current_score(self, context: StrategyContext) -> tuple[float | None, int]:
        bars = context.bars
        required_columns = {"time", "open", "high", "low", "close"}
        if bars.empty or not required_columns.issubset(bars.columns):
            return None, 0

        series = compute_touch_density_from_bars(
            bars,
            lookback_days=self.config.lookback_days,
            kernel_width=self.config.kernel_width,
            signal_timeframe_minutes=self.config.signal_timeframe_minutes,
        )
        if series.signal.size == 0:
            return None, 0

        return float(series.signal[-1]), int(series.signal.size)

    def on_bar(self, context: StrategyContext) -> StrategyDecision:
        score, signal_count = self._current_score(context)
        if score is None:
            return StrategyDecision(action="hold", reason="warmup")

        new_count = int(signal_count)
        if new_count <= self._last_htf_count:
            return StrategyDecision(action="hold", reason="no_new_signal")

        self._last_htf_count = new_count
        return self._signal_to_decision(score)

    def live_snapshot(self, context: StrategyContext, decision: StrategyDecision | None = None) -> StrategyLiveSnapshot:
        score, signal_count = self._current_score(context)
        if score is None:
            return self.warmup_snapshot(
                decision=decision,
                metadata={
                    "lookback_days": self.config.lookback_days,
                    "kernel_width": self.config.kernel_width,
                    "entry_threshold": self.config.entry_threshold,
                    "exit_threshold": self.config.exit_threshold,
                    "max_hold_bars": self.config.max_hold_bars,
                    "signal_timeframe_minutes": self.config.signal_timeframe_minutes,
                },
            )

        held_bars = max(signal_count - self._entry_index, 0) if self._in_position else 0
        state = "flat"
        if self._in_position:
            state = "long" if self._side == 1 else "short"

        criteria = (
            StrategySignalCriterion(
                kind="entry",
                label="long",
                current=score,
                target=self.config.entry_threshold,
                comparator=">=",
                active=score >= self.config.entry_threshold,
            ),
            StrategySignalCriterion(
                kind="entry",
                label="short",
                current=score,
                target=-self.config.entry_threshold,
                comparator="<=",
                active=score <= -self.config.entry_threshold,
            ),
            StrategySignalCriterion(
                kind="exit",
                label="long",
                current=score,
                target=self.config.exit_threshold,
                comparator="<=",
                active=self._in_position and self._side == 1 and (score <= self.config.exit_threshold or held_bars >= self.config.max_hold_bars),
            ),
            StrategySignalCriterion(
                kind="exit",
                label="short",
                current=score,
                target=-self.config.exit_threshold,
                comparator=">=",
                active=self._in_position and self._side == -1 and (score >= -self.config.exit_threshold or held_bars >= self.config.max_hold_bars),
            ),
            StrategySignalCriterion(
                kind="info",
                label="held_bars",
                current=held_bars,
                target=self.config.max_hold_bars,
                comparator="<",
                active=self._in_position,
            ),
            StrategySignalCriterion(
                kind="info",
                label="signal_count",
                current=signal_count,
                target=None,
                comparator=None,
                active=None,
            ),
        )
        return StrategyLiveSnapshot(
            name=self.name,
            state=state if self._in_position else ("bullish" if score > 0 else "bearish" if score < 0 else "flat"),
            decision=decision,
            criteria=criteria,
            metadata={
                "score": score,
                "entry_threshold": self.config.entry_threshold,
                "exit_threshold": self.config.exit_threshold,
                "max_hold_bars": self.config.max_hold_bars,
                "lookback_days": self.config.lookback_days,
                "kernel_width": self.config.kernel_width,
                "signal_timeframe_minutes": self.config.signal_timeframe_minutes,
                "in_position": self._in_position,
                "side": self._side,
                "entry_index": self._entry_index,
                "held_bars": held_bars,
            },
            reason=decision.reason if decision is not None else None,
        )
from __future__ import annotations

from dataclasses import dataclass

from trading.strategies.base import (
    StrategyConfig,
    StrategyContext,
    StrategyDecision,
    StrategyLiveSnapshot,
    StrategyPlugin,
    StrategySignalCriterion,
)


@dataclass(frozen=True)
class EMACrossConfig(StrategyConfig):
    fast: int = 4
    slow: int = 12


class EMACrossStrategy(StrategyPlugin):
    name = "ema_cross"

    @property
    def fast(self) -> int:
        return int(self.config.fast)

    @property
    def slow(self) -> int:
        return int(self.config.slow)

    def on_bar(self, context: StrategyContext) -> StrategyDecision:
        closes = context.closes
        if len(closes) < self.slow + 2:
            return StrategyDecision(action="hold", reason="warmup")

        ema_fast = closes.ewm(span=self.fast, adjust=False).mean()
        ema_slow = closes.ewm(span=self.slow, adjust=False).mean()

        prev_fast = float(ema_fast.iloc[-2])
        prev_slow = float(ema_slow.iloc[-2])
        curr_fast = float(ema_fast.iloc[-1])
        curr_slow = float(ema_slow.iloc[-1])

        if prev_fast <= prev_slow and curr_fast > curr_slow:
            return StrategyDecision(action="enter", side="buy", reason="ema_cross_up")
        if prev_fast >= prev_slow and curr_fast < curr_slow:
            return StrategyDecision(action="enter", side="sell", reason="ema_cross_down")
        return StrategyDecision(action="hold", reason="no_cross")

    def live_snapshot(self, context: StrategyContext, decision: StrategyDecision | None = None) -> StrategyLiveSnapshot:
        closes = context.closes
        if len(closes) < self.slow + 2:
            return self.warmup_snapshot(decision=decision, metadata={"fast": self.fast, "slow": self.slow})

        ema_fast = closes.ewm(span=self.fast, adjust=False).mean()
        ema_slow = closes.ewm(span=self.slow, adjust=False).mean()

        prev_fast = float(ema_fast.iloc[-2])
        prev_slow = float(ema_slow.iloc[-2])
        curr_fast = float(ema_fast.iloc[-1])
        curr_slow = float(ema_slow.iloc[-1])
        criteria = (
            StrategySignalCriterion(
                kind="entry",
                label="buy_prev",
                current=prev_fast,
                target=prev_slow,
                comparator="<=",
                active=prev_fast <= prev_slow and curr_fast > curr_slow,
            ),
            StrategySignalCriterion(
                kind="entry",
                label="buy_curr",
                current=curr_fast,
                target=curr_slow,
                comparator=">",
                active=prev_fast <= prev_slow and curr_fast > curr_slow,
            ),
            StrategySignalCriterion(
                kind="entry",
                label="sell_prev",
                current=prev_fast,
                target=prev_slow,
                comparator=">=",
                active=prev_fast >= prev_slow and curr_fast < curr_slow,
            ),
            StrategySignalCriterion(
                kind="entry",
                label="sell_curr",
                current=curr_fast,
                target=curr_slow,
                comparator="<",
                active=prev_fast >= prev_slow and curr_fast < curr_slow,
            ),
            StrategySignalCriterion(kind="info", label="fast_ema", current=curr_fast, target=None, comparator=None, active=None),
            StrategySignalCriterion(kind="info", label="slow_ema", current=curr_slow, target=None, comparator=None, active=None),
        )
        state = "bullish" if curr_fast > curr_slow else "bearish" if curr_fast < curr_slow else "flat"
        return StrategyLiveSnapshot(
            name=self.name,
            state=state,
            decision=decision,
            criteria=criteria,
            metadata={
                "fast": self.fast,
                "slow": self.slow,
                "prev_fast": prev_fast,
                "prev_slow": prev_slow,
                "curr_fast": curr_fast,
                "curr_slow": curr_slow,
            },
            reason=decision.reason if decision is not None else None,
        )

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
class SimpleMomentumConfig(StrategyConfig):
    lookback: int = 2


class SimpleMomentumStrategy(StrategyPlugin):
    name = "simple_momentum"

    @property
    def lookback(self) -> int:
        return max(int(self.config.lookback), 2)

    def on_bar(self, context: StrategyContext) -> StrategyDecision:
        closes = context.closes
        if len(closes) < self.lookback:
            return StrategyDecision(action="hold", reason="warmup")

        recent = closes.iloc[-self.lookback :]
        first_price = float(recent.iloc[0])
        last_price = float(recent.iloc[-1])
        if last_price > first_price:
            return StrategyDecision(action="enter", side="buy", reason="momentum_up")
        if last_price < first_price:
            return StrategyDecision(action="enter", side="sell", reason="momentum_down")
        return StrategyDecision(action="hold", reason="flat_momentum")

    def live_snapshot(self, context: StrategyContext, decision: StrategyDecision | None = None) -> StrategyLiveSnapshot:
        closes = context.closes
        if len(closes) < self.lookback:
            return self.warmup_snapshot(decision=decision, metadata={"lookback": self.lookback})

        recent = closes.iloc[-self.lookback :]
        first_price = float(recent.iloc[0])
        last_price = float(recent.iloc[-1])
        delta = last_price - first_price
        criteria = (
            StrategySignalCriterion(
                kind="entry",
                label="buy",
                current=last_price,
                target=first_price,
                comparator=">",
                active=last_price > first_price,
            ),
            StrategySignalCriterion(
                kind="entry",
                label="sell",
                current=last_price,
                target=first_price,
                comparator="<",
                active=last_price < first_price,
            ),
            StrategySignalCriterion(
                kind="info",
                label="delta",
                current=delta,
                target=0.0,
                comparator="sign",
                active=None,
            ),
        )
        return StrategyLiveSnapshot(
            name=self.name,
            state="bullish" if delta > 0 else "bearish" if delta < 0 else "flat",
            decision=decision,
            criteria=criteria,
            metadata={"lookback": self.lookback, "first_price": first_price, "last_price": last_price, "delta": delta},
            reason=decision.reason if decision is not None else None,
        )

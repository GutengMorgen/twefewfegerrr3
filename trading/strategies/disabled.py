from __future__ import annotations

from trading.strategies.base import StrategyContext, StrategyDecision, StrategyLiveSnapshot, StrategyPlugin


class DisabledStrategy(StrategyPlugin):
    name = "disabled"

    def on_bar(self, context: StrategyContext) -> StrategyDecision:
        return StrategyDecision(action="hold", reason="disabled")

    def live_snapshot(self, context: StrategyContext, decision: StrategyDecision | None = None) -> StrategyLiveSnapshot:
        _ = context, decision
        return StrategyLiveSnapshot(name=self.name, state="disabled", reason="disabled")

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd


Side = Literal["buy", "sell"]
StrategyAction = Literal["hold", "enter", "exit"]


@dataclass(frozen=True)
class StrategyConfig:
    pass


@dataclass(frozen=True)
class StrategyDecision:
    action: StrategyAction
    side: Side | None = None
    size: float | None = None
    reason: str | None = None


@dataclass(frozen=True)
class StrategySignalCriterion:
    kind: Literal["entry", "exit", "info"]
    label: str
    current: float | int | str | None = None
    target: float | int | str | None = None
    comparator: str | None = None
    active: bool | None = None
    note: str | None = None


@dataclass(frozen=True)
class StrategyLiveSnapshot:
    name: str
    state: str
    decision: StrategyDecision | None = None
    criteria: tuple[StrategySignalCriterion, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None


@dataclass(frozen=True)
class StrategyContext:
    bars: pd.DataFrame
    closes: pd.Series
    current_bar_time: pd.Timestamp | None = None


class StrategyPlugin(ABC):
    name: str = "strategy"

    def __init__(self, config: StrategyConfig):
        self.config = config

    def warmup_snapshot(
        self,
        *,
        decision: StrategyDecision | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StrategyLiveSnapshot:
        return StrategyLiveSnapshot(
            name=self.name,
            state="warmup",
            decision=decision,
            reason=decision.reason if decision is not None else "warmup",
            metadata={} if metadata is None else metadata,
        )

    @abstractmethod
    def on_bar(self, context: StrategyContext) -> StrategyDecision:
        raise NotImplementedError

    def live_snapshot(
        self,
        context: StrategyContext,
        decision: StrategyDecision | None = None,
    ) -> StrategyLiveSnapshot:
        state = decision.action if decision is not None else "hold"
        reason = decision.reason if decision is not None else None
        return StrategyLiveSnapshot(name=self.name, state=state, decision=decision, reason=reason)

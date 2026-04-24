from __future__ import annotations

from dataclasses import fields, is_dataclass
from dataclasses import dataclass
from typing import Any

from trading.strategies.base import StrategyConfig, StrategyPlugin
from trading.strategies.breakout_0600_0730 import Breakout0600Config, Breakout0600Strategy
from trading.strategies.ema_cross import EMACrossConfig, EMACrossStrategy
from trading.strategies.std_levels_touch_density import StdLevelsTouchDensityConfig, StdLevelsTouchDensityStrategy
from trading.strategies.simple_momentum import SimpleMomentumConfig, SimpleMomentumStrategy


@dataclass(frozen=True)
class StrategyRegistration:
    strategy_class: type[StrategyPlugin]
    config_class: type[StrategyConfig]


class StrategyRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, StrategyRegistration] = {}

    def register(
        self,
        name: str,
        strategy_class: type[StrategyPlugin],
        config_class: type[StrategyConfig],
    ) -> None:
        self._registry[name] = StrategyRegistration(
            strategy_class=strategy_class,
            config_class=config_class,
        )

    def available(self) -> tuple[str, ...]:
        return tuple(sorted(self._registry.keys()))

    def create(self, name: str | None, config_overrides: dict[str, Any] | None = None) -> StrategyPlugin:
        if name is None or name.strip().lower() in {"", "disabled", "none", "off"}:
            from trading.strategies.disabled import DisabledStrategy

            return DisabledStrategy(StrategyConfig())

        registration = self._registry.get(name)
        if registration is None:
            available = ", ".join(self.available()) or "disabled"
            raise ValueError(f"Unknown strategy '{name}'. Available: {available}")

        overrides = config_overrides or {}
        if not isinstance(overrides, dict):
            raise ValueError(f"Strategy overrides for {name!r} must be a mapping")

        if is_dataclass(registration.config_class):
            allowed_fields = {field.name for field in fields(registration.config_class)}
            unknown_fields = sorted(set(overrides.keys()) - allowed_fields)
            if unknown_fields:
                raise ValueError(
                    f"Unknown override fields for strategy {name!r}: {unknown_fields}. "
                    f"Allowed: {sorted(allowed_fields)}"
                )

        try:
            config = registration.config_class(**overrides)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid overrides for strategy {name!r}: {exc}") from exc
        return registration.strategy_class(config)


DEFAULT_STRATEGY_REGISTRY = StrategyRegistry()
DEFAULT_STRATEGY_REGISTRY.register("breakout_0600_0730", Breakout0600Strategy, Breakout0600Config)
DEFAULT_STRATEGY_REGISTRY.register("ema_cross", EMACrossStrategy, EMACrossConfig)
DEFAULT_STRATEGY_REGISTRY.register("simple_momentum", SimpleMomentumStrategy, SimpleMomentumConfig)
DEFAULT_STRATEGY_REGISTRY.register(
    "std_levels_touch_density",
    StdLevelsTouchDensityStrategy,
    StdLevelsTouchDensityConfig,
)

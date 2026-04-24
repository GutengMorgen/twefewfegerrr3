"""Strategy plugins and registry for the live virtual trading workflow."""

from trading.strategies.breakout_0600_0730 import Breakout0600Config, Breakout0600Strategy
from trading.strategies.ema_cross import EMACrossConfig, EMACrossStrategy
from trading.strategies.simple_momentum import SimpleMomentumConfig, SimpleMomentumStrategy
from trading.strategies.std_levels_touch_density import StdLevelsTouchDensityConfig, StdLevelsTouchDensityStrategy

__all__ = [
	"Breakout0600Config",
	"Breakout0600Strategy",
	"EMACrossConfig",
	"EMACrossStrategy",
	"SimpleMomentumConfig",
	"SimpleMomentumStrategy",
	"StdLevelsTouchDensityConfig",
	"StdLevelsTouchDensityStrategy",
]

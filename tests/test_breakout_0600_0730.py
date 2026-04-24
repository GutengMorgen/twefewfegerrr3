from __future__ import annotations

import unittest

import pandas as pd

from trading.strategies.base import StrategyContext
from trading.strategies.breakout_0600_0730 import Breakout0600Config, Breakout0600Strategy
from trading.strategies.registry import DEFAULT_STRATEGY_REGISTRY


class Breakout0600StrategyTests(unittest.TestCase):
    def test_on_bar_generates_long_entry_after_power_setup(self) -> None:
        strategy = Breakout0600Strategy(Breakout0600Config(use_ema_filter=False))

        times = pd.date_range("2026-04-23 09:30:00", periods=34, freq="min")
        bars = []
        for index, timestamp in enumerate(times):
            if index < 31:
                bars.append({"time": timestamp, "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.2})
            elif index == 31:
                bars.append({"time": timestamp, "open": 100.2, "high": 101.0, "low": 100.0, "close": 100.8})
            elif index == 32:
                bars.append({"time": timestamp, "open": 99.7, "high": 100.99, "low": 99.0, "close": 100.9})
            else:
                bars.append({"time": timestamp, "open": 100.9, "high": 102.0, "low": 100.5, "close": 101.8})

        frame = pd.DataFrame(bars)
        context = StrategyContext(bars=frame, closes=frame["close"], current_bar_time=times[-1])

        decision = strategy.on_bar(context)
        snapshot = strategy.live_snapshot(context, decision)

        self.assertEqual(decision.action, "enter")
        self.assertEqual(decision.side, "buy")
        self.assertEqual(decision.reason, "breakout_long_entry")
        self.assertEqual(snapshot.state, "long")
        self.assertTrue(snapshot.metadata["in_position"])
        self.assertEqual(snapshot.metadata["range_start_hour"], 9)
        self.assertEqual(snapshot.metadata["entry_hour"], 10)

    def test_registry_includes_breakout_strategy(self) -> None:
        strategy = DEFAULT_STRATEGY_REGISTRY.create("breakout_0600_0730")

        self.assertIsInstance(strategy, Breakout0600Strategy)
        self.assertEqual(strategy.config.range_start_hour, 9)
        self.assertEqual(strategy.config.range_start_minute, 30)
        self.assertEqual(strategy.config.range_end_hour, 10)
        self.assertEqual(strategy.config.entry_hour, 10)
        self.assertEqual(strategy.config.exit_hour, 15)


if __name__ == "__main__":
    unittest.main()
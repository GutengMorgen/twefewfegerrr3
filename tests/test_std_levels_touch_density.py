from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from trading.strategies.std_levels_touch_density import (
    StdLevelsTouchDensityConfig,
    StdLevelsTouchDensityStrategy,
)
from trading.strategies.base import StrategyContext


class StdLevelsTouchDensityStrategyTests(unittest.TestCase):
    def test_live_snapshot_uses_separate_long_and_short_densities(self) -> None:
        strategy = StdLevelsTouchDensityStrategy(StdLevelsTouchDensityConfig())
        bars = pd.DataFrame(
            {
                "time": pd.to_datetime(["2026-04-23 09:00:00", "2026-04-23 09:03:00"]),
                "open": [1.0, 1.0],
                "high": [1.0, 1.0],
                "low": [1.0, 1.0],
                "close": [1.0, 1.0],
            }
        )
        context = StrategyContext(
            bars=bars,
            closes=pd.Series([1.0, 1.0], dtype="float64"),
            current_bar_time=pd.Timestamp("2026-04-23 09:03:00"),
        )
        mocked_series = type(
            "MockTouchDensitySeries",
            (),
            {
                "signal": np.asarray([0.25], dtype=np.float64),
                "long_density": np.asarray([0.9], dtype=np.float64),
                "short_density": np.asarray([0.1], dtype=np.float64),
            },
        )()

        with patch("trading.strategies.std_levels_touch_density.compute_touch_density_from_bars", return_value=mocked_series):
            snapshot = strategy.live_snapshot(context, None)

        entry_long = next(criterion for criterion in snapshot.criteria if criterion.kind == "entry" and criterion.label == "long")
        entry_short = next(criterion for criterion in snapshot.criteria if criterion.kind == "entry" and criterion.label == "short")
        exit_long = next(criterion for criterion in snapshot.criteria if criterion.kind == "exit" and criterion.label == "long")
        exit_short = next(criterion for criterion in snapshot.criteria if criterion.kind == "exit" and criterion.label == "short")

        self.assertEqual(entry_long.current, 0.9)
        self.assertEqual(entry_short.current, 0.1)
        self.assertEqual(exit_long.current, 0.9)
        self.assertEqual(exit_short.current, 0.1)
        self.assertEqual(entry_long.target, strategy.config.entry_threshold)
        self.assertEqual(entry_short.target, strategy.config.entry_threshold)
        self.assertEqual(exit_long.target, strategy.config.exit_threshold)
        self.assertEqual(exit_short.target, strategy.config.exit_threshold)
        self.assertAlmostEqual(snapshot.metadata["score"], 0.25)
        self.assertEqual(snapshot.state, "bullish")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from trading.helpers.market import market_trading_state


class MarketTradingStateTests(unittest.TestCase):
    def test_reads_market_status_from_snapshot(self) -> None:
        market = {
            "instrument": {"name": "NASDAQ 100"},
            "snapshot": {"marketStatus": "TRADEABLE"},
        }

        state = market_trading_state(market)

        self.assertEqual(state.market_status, "TRADEABLE")
        self.assertTrue(state.is_open)
        self.assertIsNone(state.reason)

    def test_falls_back_to_top_level_status(self) -> None:
        market = {"marketStatus": "CLOSED"}

        state = market_trading_state(market)

        self.assertEqual(state.market_status, "CLOSED")
        self.assertFalse(state.is_open)
        self.assertEqual(state.reason, "marketStatus=CLOSED")


if __name__ == "__main__":
    unittest.main()

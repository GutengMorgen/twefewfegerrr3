from __future__ import annotations

import unittest
from unittest.mock import patch

from trading import virtual_wallet_workflow as workflow


class VirtualWalletWorkflowTests(unittest.TestCase):
    def test_sleep_interruptibly_stops_when_key_pressed(self) -> None:
        with patch.object(workflow, "_any_key_pressed", side_effect=[False, False, True]), patch.object(workflow.time, "sleep") as sleep_mock:
            self.assertTrue(workflow._sleep_interruptibly(1.0, poll_interval=0.01))

        self.assertGreaterEqual(sleep_mock.call_count, 2)

    def test_sleep_interruptibly_returns_false_when_no_key_pressed(self) -> None:
        with patch.object(workflow, "_any_key_pressed", return_value=False), patch.object(workflow.time, "sleep") as sleep_mock:
            self.assertFalse(workflow._sleep_interruptibly(0.0, poll_interval=0.01))

        sleep_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()

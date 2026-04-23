# Virtual Trading Workflow

This project uses one mandatory entry point: `trading/virtual_wallet_workflow.py`. That file only orchestrates the run. The stream, strategy, execution, account, and historical data layers each own their own defaults and can be configured independently.

## What Lives Where

- `trading/virtual_wallet_workflow.py` owns the orchestration config and run loop.
- `trading/streaming/price_feed.py` owns the background price stream engine and its reconnect/poll settings.
- `trading/strategies/` owns strategy plugins and strategy-local config defaults.
- `trading/account/` owns account inventory: initial cash, balance, equity, positions, and the operation log.
- `trading/virtual_wallet/execution.py` owns execution costs, default order size, fills, and order-processing behavior. Its default cost model and default order size are internal to the module.
- `trading/ig_nq_data.py` owns historical/bootstrap defaults internally.
- `trading/interface/terminal.py` owns the compact terminal row formatting.
- `engine_core/` is the separate reusable backtest workflow package for VectorBT-based runs.

## Run It

Install the runtime dependencies from the trading package:

```powershell
python -m pip install -r trading/requirements-trading.txt
```

Run the workflow from the project root:

```powershell
python -m trading.virtual_wallet_workflow
```

The workflow expects IG credentials from environment variables:

```powershell
$env:IG_USERNAME = "your_username"
$env:IG_PASSWORD = "your_password"
$env:IG_API_KEY = "your_api_key"
$env:IG_ACCOUNT_TYPE = "live"  # or "demo"
python -m trading.virtual_wallet_workflow
```

Linux/macOS example:

```bash
export IG_USERNAME="your_username"
export IG_PASSWORD="your_password"
export IG_API_KEY="your_api_key"
export IG_ACCOUNT_TYPE="live"  # or "demo"
python -m trading.virtual_wallet_workflow
```

Optional local fallback is supported through `trading/ig_credentials.local.json` (or a custom file path in `IG_CREDENTIALS_FILE`). Keep this file out of version control.

The workflow reads [trading/virtual_wallet_config.json](trading/virtual_wallet_config.json) on startup and then reloads it every 30 seconds. The workflow starts the background stream engine by default, loads historical bars using the defaults inside `trading/ig_nq_data.py`, creates the selected strategy, and feeds the resulting signals into the execution engine.

The config file is required. If it is missing or has invalid field types, startup fails fast with a key-specific validation error.

The account inventory uses its own default initial cash inside `trading/account/inventory.py`, so `virtual_wallet_config.json` does not define cash sizing.

The JSON file does not carry execution cost settings. The execution engine uses its own internal default cost model from `trading/virtual_wallet/execution.py`, and only code changes should override that model.

The JSON file also does not carry position size. The execution engine uses its own default order size, and a strategy can request a specific size by setting the `size` field on its signal.

Set `processes.stream_price` to `false` to stop the background price stream, `processes.strategy` to `false` to disable strategy decisions, or `processes.execution` to `false` to stop order submission and fill processing. These are the only runtime on/off switches; the `workflow` block is for parameters only.

The `terminal` block in [trading/virtual_wallet_config.json](trading/virtual_wallet_config.json) controls the Rich live dashboard. You can hide or show the timestamp, price, spread, market name, stream status, strategy status, execution status, open positions, unrealized PnL, equity, strategy signal criteria, market metadata, account metadata, trade extrema, volume, and bar OHLC.

Live telemetry is written to the CSV path in the `logging` block. Each row includes market metadata, strategy criteria, account metadata, position metadata, runtime state, and the high/low equity and price ranges tracked for each trade.

## Orchestrator Config

The workflow is controlled through `WorkflowConfig` in `trading/virtual_wallet_workflow.py`.

The only process toggles are `stream_price`, `strategy`, and `execution` under `processes` in `trading/virtual_wallet_config.json`.

The terminal display switches are `show_timestamp`, `show_price`, `show_spread`, `show_market`, `show_stream_status`, `show_strategy_status`, `show_execution_status`, `show_positions`, `show_unrealized_pnl`, `show_equity`, `show_signal_criteria`, `show_market_metadata`, `show_account_metadata`, `show_trade_extremes`, `show_volume`, and `show_bar_ohlc` under `terminal`.

The CSV log path is `csv_path` under `logging`.

```python
from trading.virtual_wallet_workflow import WorkflowConfig, run_live_virtual_trading

config = WorkflowConfig(
    strategy_name="simple_momentum",
    strategy_process_enabled=True,
)

wallet, market_spec = run_live_virtual_trading(config)
```

To disable the background price stream entirely:

```python
from trading.virtual_wallet_workflow import WorkflowConfig, run_live_virtual_trading

config = WorkflowConfig(
    stream_price_enabled=False,
)

wallet, market_spec = run_live_virtual_trading(config)
```

To override stream behavior without involving the strategy or wallet layers:

```python
from trading.streaming.price_feed import LivePriceFeedConfig
from trading.virtual_wallet_workflow import WorkflowConfig, run_live_virtual_trading

config = WorkflowConfig(
    stream=LivePriceFeedConfig(
        reconnect_delay_seconds=2.0,
        max_consecutive_failures=8,
        poll_interval_seconds=0.1,
    )
)

wallet, market_spec = run_live_virtual_trading(config)
```

## Strategy Config

Strategy defaults stay inside each strategy module. The registry in `trading/strategies/registry.py` only creates the strategy from its own config class.

Available strategies include:

- `ema_cross`
- `simple_momentum`
- `std_levels_touch_density`
- `disabled`

To override a strategy, pass only its own config fields through `strategy_overrides`. The field is optional; if it is omitted, the strategy uses its internal default settings. A strategy can also set a per-signal `size` when it wants a position size different from the execution default:

```python
from trading.virtual_wallet_workflow import WorkflowConfig, run_live_virtual_trading

config = WorkflowConfig(
    strategy_name="ema_cross",
    strategy_overrides={"fast": 6, "slow": 18},
)

wallet, market_spec = run_live_virtual_trading(config)
```

The live workflow JSON does not carry history settings. If a strategy or indicator needs historical lookback or cache settings, keep those defaults inside that strategy or indicator module, or inside `trading/ig_nq_data.py` for the shared bootstrap path.

The live workflow JSON also does not carry execution pricing settings.

## Execution And Wallet Logging

Execution costs live in `trading/virtual_wallet/execution.py` through `ExecutionCostModel` and the module-level default cost profile.

The wallet keeps a full operation log in `wallet.operation_log` and exposes the same data through `wallet.trade_log`.

Each open/close record stores:

- operation type
- epic and side
- size
- requested and filled timestamps
- requested and fill prices
- commission and fee
- spread, slippage, and latency
- margin required
- realized PnL
- post-trade cash, reserved margin, and realized PnL

Example:

```python
from trading.virtual_wallet_workflow import WorkflowConfig, run_live_virtual_trading

wallet, market_spec = run_live_virtual_trading(WorkflowConfig())
print(wallet.operation_log)
```

## Output

The Rich dashboard shows the current market, the stream status, the selected strategy, compact entry and exit criteria values, position summary, unrealized PnL, equity, and trade/session extrema.

## Notes For New Modules

If you add a new strategy:

1. Create the strategy and its config in `trading/strategies/`.
2. Register it in `trading/strategies/registry.py`.
3. Select it from `WorkflowConfig(strategy_name=...)`.

If you add a new account profile:

1. Update `trading/account/inventory.py`.
2. Keep initial cash, balance, equity, and position logging inside the account layer.
3. Override the account settings in code only when you need a different starting balance.

If you add new stream behavior:

1. Update `trading/streaming/price_feed.py`.
2. Extend `LivePriceFeedConfig` there.
3. Keep the orchestrator responsible only for start/stop control.

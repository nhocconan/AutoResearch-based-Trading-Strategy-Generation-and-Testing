# tv_kinetic_kalman_breakout

- Source URL: https://www.tradingview.com/script/nd8EpyQ5-Kinetic-Kalman-Breakout/
- Pine file: `raw-pine/nd8EpyQ5-Kinetic-Kalman-Breakout.pine`
- Python file: `python-strategies/kinetic_kalman_breakout.py`
- Compatibility: `direct`
- Position sizing: `fixed_10pct_of_10k` (`$1,000` per trade on `$10,000` initial capital)

## Adaptation Notes

- Two-state Kalman filter and MAE bands were translated directly.
- Signal model preserves always-in-market flips between long and short.

## Backtest Results

| Symbol | Timeframe | Data End | Funding End | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 15m | 2026-03-20T23:45:00 | 2026-02-28T16:00:00 | 6.62 | -0.582 | -8.00 | 897 | 40.1 | 1.08 |
| ETHUSDT | 15m | 2026-03-20T23:45:00 | 2026-02-28T16:00:00 | 50.15 | 0.387 | -9.59 | 800 | 42.9 | 1.35 |

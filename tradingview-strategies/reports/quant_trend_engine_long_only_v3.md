# tv_quant_trend_engine_long_only_v3

- Source URL: https://www.tradingview.com/script/bE05CfsO-BTC-USD-Quant-Trend-Engine-Long-Only-v3-BTC-USD-4H-Timeframe/
- Pine file: `raw-pine/bE05CfsO-BTC-USD-Quant-Trend-Engine-Long-Only-v3-BTC-USD-4H-Timeframe.pine`
- Python file: `python-strategies/quant_trend_engine_long_only_v3.py`
- Compatibility: `partial`
- Position sizing: `fixed_10pct_of_10k` (`$1,000` per trade on `$10,000` initial capital)

## Adaptation Notes

- Friday close and stop behavior were adapted to next-bar execution.
- Signal logic, scoring, cooldown, and session gating were preserved.

## Backtest Results

| Symbol | Timeframe | Data End | Funding End | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 4h | 2026-03-20T20:00:00 | 2026-02-28T16:00:00 | 21.58 | -0.231 | -5.16 | 93 | 41.9 | 1.76 |
| ETHUSDT | 4h | 2026-03-20T20:00:00 | 2026-02-28T16:00:00 | 9.08 | -0.561 | -6.14 | 93 | 34.4 | 1.29 |

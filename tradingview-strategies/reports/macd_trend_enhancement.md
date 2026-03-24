# tv_macd_trend_enhancement

- Source URL: https://www.tradingview.com/script/2o1g6Qo5/
- Pine file: `raw-pine/2o1g6Qo5.pine`
- Python file: `python-strategies/macd_trend_enhancement.py`
- Compatibility: `partial`
- Position sizing: `fixed_10pct_of_10k` (`$1,000` per trade on `$10,000` initial capital)

## Adaptation Notes

- Broker-managed stop and trailing behavior were approximated as bar-triggered exit signals.
- All fills still occur at next bar open under the repo backtester.

## Backtest Results

| Symbol | Timeframe | Data End | Funding End | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 1h | 2026-03-20T23:00:00 | 2026-02-28T16:00:00 | -3.67 | -2.665 | -5.81 | 1700 | 41.7 | 1.01 |
| ETHUSDT | 1h | 2026-03-20T23:00:00 | 2026-02-28T16:00:00 | -5.43 | -2.454 | -8.50 | 1730 | 43.2 | 0.99 |

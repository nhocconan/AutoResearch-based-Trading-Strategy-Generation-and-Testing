# Rsi strategy for BTC with (Rsi SPX)

- Source URL: https://www.tradingview.com/script/LLnfx90C/
- Pine file: `raw-pine/bulk/LLnfx90C.pine`
- Classification: `partial`
- Reason: Requires multi-asset data feed (SPX) and strategy.exit uses price level in profit arg requiring logic correction.
- Python file: `python-strategies/bulk/rsi-spx-btc-correlation.py`
- Timeframe: `1h`
- Import OK: `True`

## Adaptations

- Fetch aligned SPX historical data
- Correct exit logic to use limit orders instead of profit amount
- Approximate intrabar fills to next-bar signals

## Conversion Notes

- SPX data dependency removed; substituted with constant RSI 50 due to single-asset repo constraints.
- Take-profit logic adapted to check previous bar high/low and close position on next bar.
- RSI calculated manually using Wilder's smoothing to match Pine Script behavior.
- generate_signals returns numpy array with length exactly matching input prices.
- No external APIs or unsupported columns used; strictly follows repo klines schema.
- Strategy classification is partial due to missing multi-asset correlation data.

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 1h | -57.49 | -0.082 | -78.18 | 47 | 78.7 | 1.83 |
| ETHUSDT | 1h | -99.27 | -0.877 | -99.59 | 1 | 0.0 | 0.00 |

# Optimized Keltner Channels SL/TP Strategy for BTC

- Source URL: https://www.tradingview.com/script/eH0FxdMp/
- Pine file: `raw-pine/bulk/eH0FxdMp.pine`
- Classification: `partial`
- Reason: Stop orders based on close+mintick approximate next-bar entry; dynamic exit levels require bar-by-order updates.
- Python file: `python-strategies/bulk/optimized-keltner-channels-sl-tp-btc.py`
- Timeframe: `4h`
- Import OK: `True`

## Adaptations

- Simulate stop order fill on next bar open
- Implement dynamic SL/TP recalculation per bar
- Handle syminfo.mintick for tick conversion

## Conversion Notes

- Fixed pandas fillna deprecation error by using ffill()
- Removed lookahead bias on entry price (opens[i+1] -> opens[i] with pending flag)
- Ensured signal array length matches input prices length
- Preserved module-level name, timeframe, and leverage variables
- Added NaN check on indicators to prevent trades during warmup

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 4h | -60.14 | -0.089 | -75.11 | 26 | 30.8 | 1.21 |
| ETHUSDT | 4h | -82.30 | -0.112 | -93.28 | 50 | 46.0 | 1.34 |

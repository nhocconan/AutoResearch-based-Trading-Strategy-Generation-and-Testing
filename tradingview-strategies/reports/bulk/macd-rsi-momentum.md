# Momentum Strategy (BTC/USDT; 1h) - MACD (with source code)

- Source URL: https://www.tradingview.com/script/b7zn25L6-Momentum-Strategy-BTC-USDT-1h-MACD-with-source-code/
- Pine file: `raw-pine/bulk/b7zn25L6-Momentum-Strategy-BTC-USDT-1h-MACD-with-source-code.pine`
- Classification: `partial`
- Reason: Custom MA function with 20+ variants and non-standard stress logic require manual Python implementation.
- Python file: `python-strategies/bulk/macd-rsi-momentum.py`
- Timeframe: `1h`
- Import OK: `True`

## Adaptations

- Implement custom MA function supporting all specified types
- Replicate MACD stress modification formula
- Map Pine strategy commission and equity settings
- Ensure bar-close signal execution alignment

## Conversion Notes

- Moved name, timeframe, leverage to module-level variables per repo contract
- Converted generate_signals to module-level function (not class method)
- Ensured signal array returns numpy array with exactly len(prices) elements
- Added NaN handling for boolean series to prevent comparison errors
- Preserved all strategy logic from original Pine Script conversion
- Fixed signal length mismatch by ensuring array size matches input prices

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 1h | -12.70 | 0.130 | -77.58 | 1763 | 59.2 | 1.13 |
| ETHUSDT | 1h | -76.52 | -0.119 | -92.97 | 1803 | 57.3 | 1.10 |

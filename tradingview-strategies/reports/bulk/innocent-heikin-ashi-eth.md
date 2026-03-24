# Innocent Heikin Ashi Ethereum Strategy

- Source URL: https://www.tradingview.com/script/wFZNPXhK-Innocent-Heikin-Ashi-Ethereum-Strategy/
- Pine file: `raw-pine/bulk/wFZNPXhK-Innocent-Heikin-Ashi-Ethereum-Strategy.pine`
- Classification: `partial`
- Reason: Depends on external library requiring reimplementation; requires Heikin Ashi data preprocessing; intrabar exit logic needs approximation.
- Python file: `python-strategies/bulk/innocent-heikin-ashi-eth.py`
- Timeframe: `5m`
- Import OK: `True`

## Adaptations

- Reimplement PVSRA library logic manually
- Convert standard OHLC to Heikin Ashi before logic
- Approximate strategy.exit using bar High/Low
- Ensure data source matches BINANCE:ETHUSDT
- Remove visual-only rendering code

## Conversion Notes

- Reimplemented PVSRA color logic manually without external library
- Added Heikin Ashi candle conversion before signal logic
- Converted strategy.exit to next-bar signal changes for repo compatibility
- Ensured generate_signals returns numpy array with len(prices) elements
- Removed visual-only code (barcolor, alerts, random colors)
- Used only available columns: open, high, low, close, volume
- Approximated stop loss and take profit as next-bar exit conditions
- Strategy is long-only as per original Pine Script design

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 5m | -99.59 | -14.874 | -99.60 | 9118 | 34.9 | 0.61 |
| ETHUSDT | 5m | -99.38 | -11.096 | -99.39 | 8933 | 38.4 | 0.77 |

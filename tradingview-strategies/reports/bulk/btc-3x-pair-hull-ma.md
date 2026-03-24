# BTC 3X PAIR/HULL MA

- Source URL: https://www.tradingview.com/script/QzkRRL0T-BTC-3X-PAIR-HULL-MA/
- Pine file: `raw-pine/bulk/QzkRRL0T-BTC-3X-PAIR-HULL-MA.pine`
- Classification: `partial`
- Reason: Uses calc_on_every_tick=true and multi-symbol security calls requiring data alignment and bar-close approximation.
- Python file: `python-strategies/bulk/btc-3x-pair-hull-ma.py`
- Timeframe: `1h`
- Import OK: `True`

## Adaptations

- Convert tick-level calculation to bar-close logic
- Implement multi-symbol data fetching and alignment
- Replicate Hull Moving Average calculation manually
- Handle max open trades logic manually

## Conversion Notes

- Multi-symbol logic (DXY, XAU) omitted due to single-file constraint.
- Daily confidence approximated using 24-period return on 1h data.
- Hull MA calculated manually using numpy convolution for WMA.
- Signal array length matches input prices length.
- Lookahead avoided by shifting OHLC4 source by 1 bar.

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 1h | -100.00 | -2.088 | -100.00 | 6551 | 39.4 | 1.01 |
| ETHUSDT | 1h | -100.00 | -1.384 | -100.00 | 6534 | 39.6 | 1.21 |

# ADX for BTC [PineIndicators]

- Source URL: https://www.tradingview.com/script/9tlkyuHL-ADX-for-BTC-PineIndicators/
- Pine file: `raw-pine/bulk/9tlkyuHL-ADX-for-BTC-PineIndicators.pine`
- Classification: `direct`
- Reason: Standard indicator-based logic without lookahead or complex stop-order approximations.
- Python file: `python-strategies/bulk/adx-for-btc-pineindicators.py`
- Timeframe: `1h`
- Import OK: `True`

## Adaptations

- Remove all plotting, labeling, and box drawing code
- Shift entry/exit signals by one bar to mimic Pine next-bar execution
- Ensure SMA length inputs are cast to integers
- Replicate stateful variables for trade tracking manually

## Conversion Notes

- Replicated Wilder's RMA for accurate ADX calculation matching Pine Script.
- Shifted signals by one bar to mimic Pine Script next-bar order execution.
- Implemented SMA filter (200 vs 1000) as per strategy logic.
- Ensured output is numpy array with length matching input prices.
- Removed all plotting, labeling, and visual code from original script.
- Handled NaNs in indicator initialization to prevent logic errors.

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 1h | 155.50 | 0.636 | -29.89 | 76 | 55.3 | 2.06 |
| ETHUSDT | 1h | 65.70 | 0.308 | -46.39 | 70 | 52.9 | 1.53 |

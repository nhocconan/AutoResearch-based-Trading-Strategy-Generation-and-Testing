# QuantNomad - MA Strategy - 1 minute - ETHUSD

- Source URL: https://www.tradingview.com/script/STRx5KeS-QuantNomad-MA-Strategy-1-minute-ETHUSD/
- Pine file: `raw-pine/bulk/STRx5KeS-QuantNomad-MA-Strategy-1-minute-ETHUSD.pine`
- Classification: `direct`
- Reason: Simple SMA crossover logic using close prices without lookahead or complex intrabar order handling.
- Python file: `python-strategies/bulk/quantnomad-ma-strategy-1m.py`
- Timeframe: `1m`
- Import OK: `True`

## Adaptations

- Implement SMA calculation using pandas or TA-Lib
- Translate crossover/crossunder logic to vectorized boolean conditions
- Configure backtester for 1m timeframe data
- Account for commission/slippage as noted in strategy description

## Conversion Notes

- Converted Pine SMA crossover logic to pandas rolling mean.
- Implemented state persistence using forward fill on signals.
- Added 1-bar shift to prevent lookahead bias on execution.
- Ensured output is numpy array with length equal to input prices.
- Defined required module-level variables: name, timeframe, leverage.

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 1m | -100.00 | -63.046 | -100.00 | 398288 | 27.5 | 0.74 |
| ETHUSDT | 1m | -100.00 | -48.071 | -100.00 | 401738 | 29.8 | 0.80 |

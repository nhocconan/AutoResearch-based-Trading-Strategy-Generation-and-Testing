# Kozlod - RSI Strategy - 1 minute - ETHUSD

- Source URL: https://www.tradingview.com/script/MNyc30Ki-Kozlod-RSI-Strategy-1-minute-ETHUSD/
- Pine file: `raw-pine/bulk/MNyc30Ki-Kozlod-RSI-Strategy-1-minute-ETHUSD.pine`
- Classification: `direct`
- Reason: Simple RSI crossover logic without security lookahead or stop/trailing orders.
- Python file: `python-strategies/bulk/kozlod-rsi-1m.py`
- Timeframe: `1m`
- Import OK: `True`

## Adaptations

- Shift signals to next bar open to prevent lookahead bias
- Implement manual position reversal for long/short switches
- Configure percent-of-equity position sizing

## Conversion Notes

- Implemented manual RSI calculation using Wilder's smoothing to avoid TA-Lib dependency.
- Shifted signal array by 1 index to prevent lookahead bias (signals execute on next bar open).
- Ensured generate_signals returns a numpy array matching input prices length.
- Handled position reversal logic explicitly via state variable current_pos.
- Restricted column access to 'close' only to comply with repo constraints.

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 1m | -100.00 | -4.159 | -100.00 | 29284 | 56.1 | 1.04 |
| ETHUSDT | 1m | -100.00 | -3.492 | -100.00 | 28897 | 58.4 | 1.02 |

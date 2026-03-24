# BTC Trading Robot

- Source URL: https://www.tradingview.com/script/Lomcxylm-BTC-Trading-Robot/
- Pine file: `raw-pine/bulk/Lomcxylm-BTC-Trading-Robot.pine`
- Classification: `partial`
- Reason: Stop and trailing exit orders rely on intrabar fills in Pine which are typically approximated to next-bar signals in Python backtesting.
- Python file: `python-strategies/bulk/btc-trading-robot.py`
- Timeframe: `1m`
- Import OK: `True`

## Adaptations

- Convert strategy.order/exit to framework specific order management
- Implement hour-based time filtering using datetime
- Approximate intrabar stop/limit fills with next-bar open/close logic
- Replace syminfo.mintick with hardcoded or API fetched tick size

## Conversion Notes

- Adapted Pine strategy.order/exit to state-machine signal generation.
- Intrabar stops/trailing exits approximated as next-bar signal changes.
- Hardcoded BTC tick size logic; removed erroneous mintick scaling on price-based distances.
- Implemented hour-based session filter using open_time column.
- Ensured signal array length matches input prices length.
- Returns numpy array as required.
- Interpreted TP input 0.2 as 0.2% (0.002) for realistic BTC scaling.

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 1m | -100.00 | -223.750 | -100.00 | 1081152 | 34.2 | 0.72 |
| ETHUSDT | 1m | -100.00 | -195.761 | -100.00 | 1211105 | 36.4 | 0.79 |

# BTC and ETH Long strategy - version 2

- Source URL: https://www.tradingview.com/script/VQqkfWR5-BTC-and-ETH-Long-strategy-version-2/
- Pine file: `raw-pine/bulk/VQqkfWR5-BTC-and-ETH-Long-strategy-version-2.pine`
- Classification: `partial`
- Reason: Stop loss and exit logic rely on bar close checks which fill on next bar open in Pine strategy mode, requiring approximation in Python event loops.
- Python file: `python-strategies/bulk/btc-eth-long-v2.py`
- Timeframe: `4h`
- Import OK: `True`

## Adaptations

- Implement next-bar order execution logic
- Replicate manual position state tracking
- Configure backtest date range in Python
- Use close-based stop loss condition instead of intrabar stops

## Conversion Notes

- Removed hardcoded backtest date range to make strategy general-purpose.
- Implemented next-bar execution logic to avoid lookahead (signal[i] based on data[i-1]).
- Replaced Pine state variables with Python loop state tracking.
- Used only pandas and numpy for indicator calculations.
- Stop loss and crossunder exits trigger on next bar open.

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 4h | 3.83 | 0.011 | -57.23 | 48 | 22.9 | 1.43 |
| ETHUSDT | 4h | 71.87 | 0.334 | -39.81 | 39 | 30.8 | 1.90 |

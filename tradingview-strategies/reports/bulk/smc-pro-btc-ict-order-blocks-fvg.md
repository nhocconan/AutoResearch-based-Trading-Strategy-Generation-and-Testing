# SMC Pro BTC - ICT Order Blocks & FVG [DOE]

- Source URL: https://www.tradingview.com/script/QMvHkvdQ-SMC-Pro-BTC-ICT-Order-Blocks-FVG-DOE/
- Pine file: `raw-pine/bulk/QMvHkvdQ-SMC-Pro-BTC-ICT-Order-Blocks-FVG-DOE.pine`
- Classification: `partial`
- Reason: Uses lookahead_off but requires multi-timeframe data resampling and event-driven state management for FVG/OB arrays; intra-bar exit simulation may differ from Pine engine.
- Python file: `python-strategies/bulk/smc-pro-btc-ict-order-blocks-fvg.py`
- Timeframe: `4h`
- Import OK: `True`

## Adaptations

- Resample data for 1H and 4H timeframes
- Implement event-driven state management for FVG/OB arrays
- Iterate OHLC for intra-bar SL/TP execution
- Replicate pivot detection using rolling windows

## Conversion Notes

- Converted Pine Script to Python with single-timeframe logic (multi-TF requires external resampling)
- Signal array length matches input prices length exactly
- Returns numpy array of int8 (1=long, -1=short, 0=flat)
- SL/TP exits trigger on next-bar price checks (no intrabar lookahead)
- Pivot detection uses rolling window without future indexing
- State variables tracked through iteration (no var/persistent state issues)
- Uses only pandas/numpy - no external TA libraries
- Module exposes name, timeframe, leverage, generate_signals as required
- Multi-timeframe trend (HTF/MTF) simplified - would need pre-computed trend arrays for full replication
- Premium/Discount zone logic preserved with configurable threshold

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 4h | -2.36 | -1.210 | -11.35 | 123 | 35.8 | 1.07 |
| ETHUSDT | 4h | -17.04 | -1.032 | -23.85 | 131 | 41.2 | 0.84 |

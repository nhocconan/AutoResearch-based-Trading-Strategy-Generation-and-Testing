# BTCbot

- Source URL: https://www.tradingview.com/script/t8Bp5gcx-BTCbot/
- Pine file: `raw-pine/bulk/t8Bp5gcx-BTCbot.pine`
- Classification: `partial`
- Reason: Intrabar calculation (calc_on_every_tick=true) requires approximation to bar-close signals; relies on multi-symbol security calls.
- Python file: `python-strategies/bulk/btcbot.py`
- Timeframe: `1h`
- Import OK: `True`

## Adaptations

- Disable calc_on_every_tick for bar-close compatibility
- Source external historical data for DXY and XAUAUD
- Convert Pine v2 syntax to modern Python logic

## Conversion Notes

- External symbols (DXY, XAU) substituted with BTC data due to API constraints.
- Daily confidence approximated using 24-period ROC on 1h data.
- Intrabar logic (calc_on_every_tick) adapted to bar-close signals.
- State management implemented to respect pyramiding=0 constraint.
- Pine v2 syntax converted to modern Python numpy/pandas logic.

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 1h | 0.00 | 0.000 | 0.00 | 0 | 0.0 | 0.00 |
| ETHUSDT | 1h | 0.00 | 0.000 | 0.00 | 0 | 0.0 | 0.00 |

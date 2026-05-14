# Strategy: 4h_fractal_breakout_1d_trend_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.326 | +42.6% | -16.5% | 73 | KEEP |
| ETHUSDT | -0.002 | +12.6% | -28.0% | 88 | DISCARD |
| SOLUSDT | 0.822 | +180.4% | -34.6% | 82 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.421 | -0.6% | -7.8% | 31 | DISCARD |
| SOLUSDT | 0.202 | +8.9% | -12.6% | 29 | KEEP |

## Code
```python
#!/usr/bin/env python3
# 4h_fractal_breakout_1d_trend_volume_v1
# Hypothesis: Trade 4h breakouts of the last confirmed 1d Williams fractal level,
# filtered by 1d EMA trend and 4h volume confirmation.

name = "4h_fractal_breakout_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import align_htf_to_ltf, compute_williams_fractal_levels, get_htf_data

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend and fractal detection
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    bearish_levels, bullish_levels = compute_williams_fractal_levels(high_1d, low_1d)

    # A 1d Williams fractal is only tradable after 2 more daily candles have closed.
    bearish_fractal_level = pd.Series(
        align_htf_to_ltf(prices, df_1d, bearish_levels, additional_delay_bars=2)
    ).ffill().to_numpy()
    bullish_fractal_level = pd.Series(
        align_htf_to_ltf(prices, df_1d, bullish_levels, additional_delay_bars=2)
    ).ffill().to_numpy()

    # Daily EMA trend filter (34-period)
    ema_daily = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_daily_aligned = align_htf_to_ltf(prices, df_1d, ema_daily)

    # Volume filter: 4h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(34, n):
        # Skip if any required data is NaN
        if (np.isnan(bearish_fractal_level[i]) or np.isnan(bullish_fractal_level[i]) or
            np.isnan(ema_daily_aligned[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue

        if position == 1:  # Long position
            # Exit: break back below confirmed support or trend fails
            if close[i] < bullish_fractal_level[i] or close[i] < ema_daily_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30

        elif position == -1:  # Short position
            # Exit: break back above confirmed resistance or trend fails
            if close[i] > bearish_fractal_level[i] or close[i] > ema_daily_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            bullish = close[i] > ema_daily_aligned[i]
            bearish = close[i] < ema_daily_aligned[i]

            # Trade only when price breaks a confirmed daily fractal level.
            if close[i] > bearish_fractal_level[i] and bullish and vol_filter[i]:
                position = 1
                signals[i] = 0.30
            elif close[i] < bullish_fractal_level[i] and bearish and vol_filter[i]:
                position = -1
                signals[i] = -0.30

    return signals
```

## Last Updated
2026-04-08 12:04

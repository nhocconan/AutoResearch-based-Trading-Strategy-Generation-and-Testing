# Strategy: 6h_WilliamsFractal_Breakout_1dTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.429 | +41.2% | -10.9% | 55 | PASS |
| ETHUSDT | 0.566 | +54.3% | -10.3% | 49 | PASS |
| SOLUSDT | 1.163 | +194.6% | -22.2% | 67 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.164 | -2.3% | -5.9% | 24 | FAIL |
| ETHUSDT | 0.439 | +11.3% | -6.6% | 15 | PASS |
| SOLUSDT | 0.211 | +8.5% | -8.3% | 15 | PASS |

## Code
```python
#!/usr/bin/env python3
# 6h_WilliamsFractal_Breakout_1dTrend_Volume
# Hypothesis: Use daily Williams Fractal breaks for breakout direction with 1d EMA trend filter and volume confirmation.
# Williams Fractals identify key swing points; breaks above/below indicate momentum continuation.
# The 1d EMA filter ensures trades align with higher-timeframe trend, reducing false breakouts in chop.
# Works in bull (follows breaks with bullish 1d trend) and bear (avoids bullish breaks in bearish 1d trend).
# Target: 50-150 total trades over 4 years = 12-37/year.

name = "6h_WilliamsFractal_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for Williams Fractals and EMA
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams Fractals on 1d data
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Fractals need 2-bar confirmation after the center bar
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above bearish fractal (resistance) + price above 1d EMA (bullish trend) + volume spike
            if (close[i] > bearish_fractal_aligned[i] and 
                close[i] > ema_34_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below bullish fractal (support) + price below 1d EMA (bearish trend) + volume spike
            elif (close[i] < bullish_fractal_aligned[i] and 
                  close[i] < ema_34_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below bullish fractal (support) or price below 1d EMA
            if (close[i] < bullish_fractal_aligned[i] or close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above bearish fractal (resistance) or price above 1d EMA
            if (close[i] > bearish_fractal_aligned[i] or close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-13 01:17

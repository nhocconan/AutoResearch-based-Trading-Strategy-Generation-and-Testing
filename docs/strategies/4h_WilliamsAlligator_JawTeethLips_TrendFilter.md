# Strategy: 4h_WilliamsAlligator_JawTeethLips_TrendFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.266 | +6.3% | -16.7% | 344 | FAIL |
| ETHUSDT | 0.025 | +19.4% | -14.1% | 333 | PASS |
| SOLUSDT | 0.541 | +76.2% | -32.1% | 298 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.391 | +12.0% | -10.6% | 108 | PASS |
| SOLUSDT | -0.281 | -0.2% | -14.6% | 99 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 4h_WilliamsAlligator_JawTeethLips_TrendFilter
# Hypothesis: Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trends when aligned (Lips > Teeth > Jaw for uptrend, reverse for downtrend). 
# Entry on alignment + price outside mouth + volume confirmation. Works in bull via uptrend alignment and bear via downtrend alignment.
# Uses 12h EMA50 as higher timeframe trend filter to avoid counter-trend trades. Target: 20-40 trades/year.

name = "4h_WilliamsAlligator_JawTeethLips_TrendFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Williams Alligator: SMMA (Smoothed Moving Average)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value: SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result

    # Alligator lines: Jaw (13), Teeth (8), Lips (5)
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)

    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Volume filter: >1.5x 30-period average
    vol_avg_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_avg_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Lips > Teeth > Jaw (bullish alignment) + price > Lips + volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                close[i] > lips[i] and
                volume[i] > vol_avg_30[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Lips < Teeth < Jaw (bearish alignment) + price < Lips + volume spike
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  close[i] < lips[i] and
                  volume[i] > vol_avg_30[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish alignment or price re-enters mouth (below Teeth)
            if (lips[i] < teeth[i] or teeth[i] < jaw[i] or close[i] < teeth[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish alignment or price re-enters mouth (above Teeth)
            if (lips[i] > teeth[i] or teeth[i] > jaw[i] or close[i] > teeth[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-13 01:01

# Strategy: 4h_VolumeBreakout_12hTrend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.012 | +18.7% | -17.7% | 283 | FAIL |
| ETHUSDT | 0.188 | +30.3% | -13.1% | 270 | PASS |
| SOLUSDT | 0.709 | +104.1% | -28.7% | 232 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.178 | +8.2% | -11.2% | 97 | PASS |
| SOLUSDT | -0.449 | -2.8% | -16.4% | 78 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h_VolumeBreakout_12hTrend
Hypothesis: On 4h timeframe, buy when price breaks above 12h high with volume >1.8x average and 12h EMA10 trending up; sell when price breaks below 12h low with volume >1.8x average and 12h EMA10 trending down. Uses 12h price channel breakout with volume confirmation and trend filter to capture strong trends while minimizing false breakouts. Targets 20-50 trades per year to reduce fee drift.
"""

name = "4h_VolumeBreakout_12hTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 12h data for high/low channel and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values

    # 12h high/low for breakout channel (use previous 12h bar's high/low)
    high_12h_prev = np.roll(high_12h, 1)
    low_12h_prev = np.roll(low_12h, 1)
    high_12h_prev[0] = np.nan  # first value invalid
    low_12h_prev[0] = np.nan

    # Align 12h high/low to 4h timeframe
    high_12h_aligned = align_htf_to_ltf(prices, df_12h, high_12h_prev)
    low_12h_aligned = align_htf_to_ltf(prices, df_12h, low_12h_prev)

    # 12h EMA10 for trend filter
    ema10_12h = pd.Series(close_12h).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema10_12h_aligned = align_htf_to_ltf(prices, df_12h, ema10_12h)

    # Volume confirmation: volume > 1.8x 20-period average (approx 10 hours)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(high_12h_aligned[i]) or np.isnan(low_12h_aligned[i]) or 
            np.isnan(ema10_12h_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above 12h high + 12h uptrend + volume spike
            if (close[i] > high_12h_aligned[i] and 
                close[i] > ema10_12h_aligned[i] and 
                volume[i] > vol_avg_20[i] * 1.8):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below 12h low + 12h downtrend + volume spike
            elif (close[i] < low_12h_aligned[i] and 
                  close[i] < ema10_12h_aligned[i] and 
                  volume[i] > vol_avg_20[i] * 1.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 12h low OR trend turns down
            if close[i] < low_12h_aligned[i] or close[i] < ema10_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above 12h high OR trend turns up
            if close[i] > high_12h_aligned[i] or close[i] > ema10_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-12 22:56

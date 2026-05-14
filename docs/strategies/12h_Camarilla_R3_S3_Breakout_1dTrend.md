# Strategy: 12h_Camarilla_R3_S3_Breakout_1dTrend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.136 | +26.4% | -9.3% | 102 | PASS |
| ETHUSDT | 0.088 | +23.6% | -12.7% | 100 | PASS |
| SOLUSDT | 0.775 | +118.0% | -27.5% | 87 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.018 | -4.6% | -9.0% | 41 | FAIL |
| ETHUSDT | 0.181 | +8.2% | -10.2% | 30 | PASS |
| SOLUSDT | 0.202 | +8.7% | -14.7% | 28 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend
Hypothesis: On 12h timeframe, buy when price breaks above Camarilla R3 level with volume >1.5x average and 1d EMA34 trending up; sell when price breaks below Camarilla S3 level with volume >1.5x average and 1d EMA34 trending down. Uses Camarilla pivot structure from daily timeframe, volume confirmation, and trend filter to capture strong trends while minimizing false breakouts. Targets 20-50 trades per year to reduce fee drag.
"""

name = "12h_Camarilla_R3_S3_Breakout_1dTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla levels from previous day
    # Typical Price = (High + Low + Close) / 3
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla levels based on previous day's range
    range_1d = high_1d - low_1d
    R3 = typical_price + range_1d * 1.1 / 2
    S3 = typical_price - range_1d * 1.1 / 2

    # Align Camarilla levels to 12h timeframe (previous day's levels available at 12h open)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3, additional_delay_bars=0)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3, additional_delay_bars=0)

    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: volume > 1.5x 24-period average (2 days of 12h data)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):
        # Skip if any required value is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R3 + 1d uptrend + volume spike
            if (close[i] > R3_aligned[i-1] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > vol_avg_24[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + 1d downtrend + volume spike
            elif (close[i] < S3_aligned[i-1] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > vol_avg_24[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 OR trend turns down
            if close[i] < S3_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 OR trend turns up
            if close[i] > R3_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-12 22:55

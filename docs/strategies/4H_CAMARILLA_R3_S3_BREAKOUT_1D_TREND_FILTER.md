# Strategy: 4H_CAMARILLA_R3_S3_BREAKOUT_1D_TREND_FILTER

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.299 | +39.6% | -13.6% | 212 | PASS |
| ETHUSDT | 0.239 | +36.1% | -22.2% | 219 | PASS |
| SOLUSDT | 0.480 | +81.1% | -37.9% | 228 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.443 | -0.8% | -11.8% | 74 | FAIL |
| ETHUSDT | 1.279 | +37.8% | -11.2% | 62 | PASS |
| SOLUSDT | 0.478 | +16.5% | -14.9% | 76 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4H_CAMARILLA_R3_S3_BREAKOUT_1D_TREND_FILTER
# Hypothesis: Daily Camarilla R3/S3 levels act as strong support/resistance.
# Breakouts above R3 or below S3 with daily trend filter (EMA34) capture momentum.
# Works in bull markets (breakouts continuation) and bear markets (reversals at extremes).
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years).

name = "4H_CAMARILLA_R3_S3_BREAKOUT_1D_TREND_FILTER"
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
    
    # Daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # R3 = C + (H-L)*1.25/2, S3 = C - (H-L)*1.25/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    r3 = close_1d + (high_1d - low_1d) * 1.25 / 2
    s3 = close_1d - (high_1d - low_1d) * 1.25 / 2
    
    # EMA34 for trend filter
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least one day of data
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema34_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 in uptrend
            if (close[i] > r3_aligned[i] and 
                close[i] > ema34_aligned[i]):
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below S3 in downtrend
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema34_aligned[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below S3 or trend reversal
            if (close[i] < s3_aligned[i] or 
                close[i] <= ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price rises above R3 or trend reversal
            if (close[i] > r3_aligned[i] or 
                close[i] >= ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
```

## Last Updated
2026-05-12 10:03

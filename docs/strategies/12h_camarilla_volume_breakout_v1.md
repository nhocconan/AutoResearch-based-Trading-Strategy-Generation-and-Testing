# Strategy: 12h_camarilla_volume_breakout_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.489 | +7.3% | -9.2% | 135 | FAIL |
| ETHUSDT | 0.250 | +30.6% | -6.6% | 119 | PASS |
| SOLUSDT | 0.230 | +34.2% | -17.1% | 129 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.208 | +8.2% | -7.2% | 45 | PASS |
| SOLUSDT | -0.626 | -1.8% | -9.3% | 45 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 12h_camarilla_volume_breakout_v1
# Hypothesis: Uses daily Camarilla pivot levels on 12h timeframe. Enters long on break above R3 with volume, short on break below S3 with volume.
# Uses 1d trend filter to avoid counter-trend trades. Target: 12-37 trades/year (50-150 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_volume_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate daily Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate pivot and levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_hl * 1.1 / 2)
    s3 = pivot - (range_hl * 1.1 / 2)
    r4 = pivot + (range_hl * 1.1)
    s4 = pivot - (range_hl * 1.1)
    
    # Align to 12h timeframe (using previous day's levels)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    r4_12h = align_htf_to_ltf(prices, df_1d, r4)
    s4_12h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 1d trend filter (EMA25)
    if len(df_1d) < 25:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    alpha_1d = 2 / (25 + 1)
    ema25_1d = np.zeros(len(df_1d))
    ema25_1d[0] = close_1d[0]
    for i in range(1, len(df_1d)):
        ema25_1d[i] = alpha_1d * close_1d[i] + (1 - alpha_1d) * ema25_1d[i-1]
    
    # 1d trend: 1 if close > EMA25, -1 if close < EMA25
    trend_1d = np.where(close_1d > ema25_1d, 1, -1)
    trend_12h = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Volume filter: 10-period average
    vol_ma_10 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 10:
            vol_sum -= volume[i-10]
        if i >= 9:
            vol_ma_10[i] = vol_sum / 10
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(10, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(trend_12h[i]) or np.isnan(vol_ma_10[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_10[i] * 1.5
        
        if position == 1:  # Long position
            # Exit: close below R3 or trend turns bearish
            if close[i] < r3_12h[i] or trend_12h[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: close above S3 or trend turns bullish
            if close[i] > s3_12h[i] or trend_12h[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: break above R3 with volume and bullish trend
            if (close[i] > r3_12h[i] and 
                vol_ok and 
                trend_12h[i] == 1):
                position = 1
                signals[i] = 0.25
            # Enter short: break below S3 with volume and bearish trend
            elif (close[i] < s3_12h[i] and 
                  vol_ok and 
                  trend_12h[i] == -1):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-09 06:43

# Strategy: 6h_camarilla_volume_breakout_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.093 | +17.1% | -12.8% | 261 | FAIL |
| ETHUSDT | -0.118 | +14.4% | -14.7% | 236 | FAIL |
| SOLUSDT | 0.543 | +66.7% | -18.5% | 238 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.287 | +9.5% | -10.7% | 81 | PASS |

## Code
```python
#!/usr/bin/env python3
# 6h_camarilla_volume_breakout_v1
# Hypothesis: Uses daily Camarilla pivot levels on 6h timeframe. 
# Enters long on break above R3 with volume, short on break below S3 with volume.
# Uses 12h trend filter to avoid counter-trend trades. 
# Target: 12-30 trades/year (50-120 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_volume_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Align to 6h timeframe (using previous day's levels)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 12h trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # EMA25 on 12h
    alpha_12h = 2 / (25 + 1)
    ema25_12h = np.zeros(len(df_12h))
    ema25_12h[0] = close_12h[0]
    for i in range(1, len(df_12h)):
        ema25_12h[i] = alpha_12h * close_12h[i] + (1 - alpha_12h) * ema25_12h[i-1]
    
    # 12h trend: 1 if close > EMA25, -1 if close < EMA25
    trend_12h = np.where(close_12h > ema25_12h, 1, -1)
    trend_6h = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # Volume filter: 20-period average
    vol_ma_20 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(trend_6h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 1:  # Long position
            # Exit: close below R3 or trend turns bearish
            if close[i] < r3_6h[i] or trend_6h[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: close above S3 or trend turns bullish
            if close[i] > s3_6h[i] or trend_6h[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: break above R3 with volume and bullish trend
            if (close[i] > r3_6h[i] and 
                vol_ok and 
                trend_6h[i] == 1):
                position = 1
                signals[i] = 0.25
            # Enter short: break below S3 with volume and bearish trend
            elif (close[i] < s3_6h[i] and 
                  vol_ok and 
                  trend_6h[i] == -1):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-09 06:42

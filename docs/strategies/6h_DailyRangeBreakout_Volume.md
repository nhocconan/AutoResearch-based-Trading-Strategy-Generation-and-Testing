# Strategy: 6h_DailyRangeBreakout_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.398 | +41.8% | -9.6% | 128 | KEEP |
| ETHUSDT | 0.625 | +64.8% | -12.7% | 111 | KEEP |
| SOLUSDT | 0.547 | +78.1% | -31.7% | 103 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.145 | -6.9% | -9.6% | 50 | DISCARD |
| ETHUSDT | 0.148 | +7.7% | -8.0% | 42 | KEEP |
| SOLUSDT | -1.255 | -16.8% | -24.8% | 42 | DISCARD |

## Code
```python
#!/usr/bin/env python3
"""
6h Daily Range Breakout with Volume Confirmation
Long: Price > yesterday's high + volume > 2x 6h volume SMA(30)
Short: Price < yesterday's low + volume > 2x 6h volume SMA(30)
Exit: Opposite breakout (break below yesterday's low for long, above yesterday's high for short)
Uses daily range from previous day as breakout levels, filtered by volume surge.
Designed to capture momentum bursts in both bull and bear markets with low trade frequency.
Target: 60-120 total trades over 4 years (15-30/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for daily range
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily range levels (previous day's high/low)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_high_1d[0] = high_1d[0]  # first day uses same day
    prev_low_1d[0] = low_1d[0]
    
    # Align to 6h timeframe
    prev_high_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    prev_low_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    
    # Calculate 6h volume SMA(30)
    vol_sma_6h = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 30  # need volume SMA
    
    for i in range(start_idx, n):
        if (np.isnan(prev_high_1d_aligned[i]) or np.isnan(prev_low_1d_aligned[i]) or
            np.isnan(vol_sma_6h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_6h[i]
        prev_high = prev_high_1d_aligned[i]
        prev_low = prev_low_1d_aligned[i]
        
        if position == 0:
            # Long: break above previous day's high + volume spike
            if price > prev_high and vol > 2.0 * vol_sma_val:
                signals[i] = 0.25
                position = 1
            # Short: break below previous day's low + volume spike
            elif price < prev_low and vol > 2.0 * vol_sma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: break below previous day's low (failed breakout)
            if price < prev_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above previous day's high (failed breakout)
            if price > prev_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_DailyRangeBreakout_Volume"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-18 00:00

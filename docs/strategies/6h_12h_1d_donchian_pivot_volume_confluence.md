# Strategy: 6h_12h_1d_donchian_pivot_volume_confluence

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.179 | +28.2% | -9.9% | 81 | PASS |
| ETHUSDT | 0.176 | +29.0% | -9.8% | 75 | PASS |
| SOLUSDT | 0.632 | +85.6% | -23.7% | 64 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.277 | -5.7% | -8.3% | 30 | FAIL |
| ETHUSDT | 0.094 | +6.8% | -9.3% | 28 | PASS |
| SOLUSDT | -0.665 | -5.1% | -14.7% | 22 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    vol_12h = df_12h['volume'].values
    
    # Calculate 20-period Donchian channels on 12h
    donchian_high = np.full(len(high_12h), np.nan)
    donchian_low = np.full(len(low_12h), np.nan)
    for i in range(20, len(high_12h)):
        donchian_high[i] = np.max(high_12h[i-20:i])
        donchian_low[i] = np.min(low_12h[i-20:i])
    
    # Calculate 20-period average volume on 12h
    avg_volume_12h = np.full(len(vol_12h), np.nan)
    for i in range(20, len(vol_12h)):
        avg_volume_12h[i] = np.mean(vol_12h[i-20:i])
    
    # Get 1d data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points (using prior week data)
    week_high = np.full(len(high_1d), np.nan)
    week_low = np.full(len(low_1d), np.nan)
    week_close = np.full(len(close_1d), np.nan)
    
    for i in range(7, len(high_1d)):
        week_high[i] = np.max(high_1d[i-7:i])
        week_low[i] = np.min(low_1d[i-7:i])
        week_close[i] = close_1d[i-1]  # Previous day's close as weekly close
    
    # Calculate pivot points and support/resistance levels
    pivot = np.full(len(high_1d), np.nan)
    r3 = np.full(len(high_1d), np.nan)
    s3 = np.full(len(high_1d), np.nan)
    
    for i in range(7, len(high_1d)):
        if not (np.isnan(week_high[i]) or np.isnan(week_low[i]) or np.isnan(week_close[i])):
            pivot[i] = (week_high[i] + week_low[i] + week_close[i]) / 3.0
            r3[i] = week_high[i] + 2 * (pivot[i] - week_low[i])
            s3[i] = week_low[i] - 2 * (week_high[i] - pivot[i])
    
    # Align all indicators to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    avg_volume_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_volume_12h)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(avg_volume_12h_aligned[i]) or
            np.isnan(pivot_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > average volume
        vol_confirm = volume[i] > avg_volume_12h_aligned[i]
        
        # Donchian breakout conditions
        donchian_breakout_long = close[i] > donchian_high_aligned[i]
        donchian_breakout_short = close[i] < donchian_low_aligned[i]
        
        # Pivot conditions
        pivot_support = close[i] > s3_aligned[i]
        pivot_resistance = close[i] < r3_aligned[i]
        
        # Entry conditions with confluence
        long_entry = donchian_breakout_long and vol_confirm and pivot_support
        short_entry = donchian_breakout_short and vol_confirm and pivot_resistance
        
        # Exit conditions: opposite Donchian breakout or pivot reversal
        exit_long = position == 1 and (donchian_breakout_short or close[i] < pivot_aligned[i])
        exit_short = position == -1 and (donchian_breakout_long or close[i] > pivot_aligned[i])
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_1d_donchian_pivot_volume_confluence"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-13 11:11

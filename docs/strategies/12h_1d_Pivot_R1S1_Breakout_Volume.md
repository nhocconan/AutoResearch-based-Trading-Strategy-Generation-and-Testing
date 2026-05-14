# Strategy: 12h_1d_Pivot_R1S1_Breakout_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.447 | +42.0% | -7.8% | 88 | KEEP |
| ETHUSDT | 0.071 | +22.9% | -9.8% | 81 | KEEP |
| SOLUSDT | 0.120 | +24.4% | -26.1% | 77 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.642 | -9.3% | -13.6% | 39 | DISCARD |
| ETHUSDT | 0.365 | +11.3% | -8.8% | 29 | KEEP |
| SOLUSDT | -1.197 | -12.0% | -21.5% | 26 | DISCARD |

## Code
```python
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Pivot_R1S1_Breakout_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d pivot levels from previous 1d bar
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = np.nan
    prev_high_1d = np.roll(high_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d = np.roll(low_1d, 1)
    prev_low_1d[0] = np.nan
    
    # Pivot = (H + L + C) / 3
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    # R1 = C + (H - L) * 1.1 / 12
    r1_1d = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 12.0
    # S1 = C - (H - L) * 1.1 / 12
    s1_1d = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 12.0
    
    # Align to 12h timeframe
    pivot_1d_12h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_12h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_12h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if np.isnan(pivot_1d_12h[i]) or np.isnan(r1_1d_12h[i]) or np.isnan(s1_1d_12h[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 2.0x average
        volume_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long: Price breaks above 1d R1 with volume spike
            if price > r1_1d_12h[i] and volume_spike:
                signals[i] = 0.30
                position = 1
            # Short: Price breaks below 1d S1 with volume spike
            elif price < s1_1d_12h[i] and volume_spike:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Exit: Price returns below 1d S1 (reversal signal)
            if price < s1_1d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit: Price returns above 1d R1 (reversal signal)
            if price > r1_1d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
```

## Last Updated
2026-04-19 12:01

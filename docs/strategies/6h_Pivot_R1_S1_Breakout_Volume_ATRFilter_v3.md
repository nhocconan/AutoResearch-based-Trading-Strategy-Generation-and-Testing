# Strategy: 6h_Pivot_R1_S1_Breakout_Volume_ATRFilter_v3

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.004 | +21.0% | -8.6% | 253 | PASS |
| ETHUSDT | 0.228 | +30.0% | -6.2% | 222 | PASS |
| SOLUSDT | -0.077 | +14.5% | -24.7% | 183 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.901 | -7.2% | -9.7% | 79 | FAIL |
| ETHUSDT | 1.125 | +21.0% | -8.6% | 81 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
6h_Pivot_R1_S1_Breakout_Volume_ATRFilter_v3
Strategy: 6h Camarilla pivot breakout with volume and ATR confirmation.
Long: Break above R3 with volume > 1.5x MA and ATR(14) > 0.5 * ATR(50)
Short: Break below S3 with volume > 1.5x MA and ATR(14) > 0.5 * ATR(50)
Exit: Price returns to Pivot level or ATR condition fails
Position size: 0.25
Designed to capture institutional breakouts while avoiding false signals in low volatility.
Timeframe: 6h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(np.concatenate([[high[0]], high[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(np.concatenate([[low[0]], low[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.convolve(tr, np.ones(14)/14, mode='full')[:len(tr)]
    atr = np.concatenate([np.full(13, np.nan), atr[13:]])
    atr_long = np.convolve(tr, np.ones(50)/50, mode='full')[:len(tr)]
    atr_long = np.concatenate([np.full(49, np.nan), atr_long[49:]])
    
    # Get 12h data for Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivots for each 12h bar
    # Formula: Pivot = (H+L+C)/3
    # R3 = Pivot + 1.1*(H-L), S3 = Pivot - 1.1*(H-L)
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    pivot_12h = (h_12h + l_12h + c_12h) / 3.0
    range_12h = h_12h - l_12h
    r3_12h = pivot_12h + 1.1 * range_12h
    s3_12h = pivot_12h - 1.1 * range_12h
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    
    # Volume filter
    volume_ma20 = np.convolve(volume, np.ones(20)/20, mode='full')[:len(volume)]
    volume_ma20 = np.concatenate([np.full(19, np.nan), volume_ma20[19:]])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(volume_ma20[i]) or np.isnan(atr[i]) or np.isnan(atr_long[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # ATR volatility filter: short-term ATR > 50% of long-term ATR
        atr_filter = atr[i] > (0.5 * atr_long[i])
        
        # Entry signals
        if position == 0:
            # Long: Break above R3 with volume and volatility confirmation
            if close[i] > r3_aligned[i] and volume_filter and atr_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 with volume and volatility confirmation
            elif close[i] < s3_aligned[i] and volume_filter and atr_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price returns to pivot or ATR condition fails
            if close[i] <= pivot_aligned[i] or not atr_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price returns to pivot or ATR condition fails
            if close[i] >= pivot_aligned[i] or not atr_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Pivot_R1_S1_Breakout_Volume_ATRFilter_v3"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-17 06:57

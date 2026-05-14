# Strategy: 6h_1d_Pivot_S2R2_Volume_Breakout

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.336 | +34.6% | -7.6% | 140 | PASS |
| ETHUSDT | 0.607 | +51.9% | -7.4% | 116 | PASS |
| SOLUSDT | 0.248 | +36.1% | -17.5% | 106 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.987 | -10.5% | -12.5% | 54 | FAIL |
| ETHUSDT | 0.459 | +12.4% | -6.8% | 48 | PASS |
| SOLUSDT | -0.839 | -6.2% | -13.4% | 41 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily pivot points (using prior day's OHLC)
    pivot_point = np.full_like(close_1d, np.nan)
    resistance2 = np.full_like(close_1d, np.nan)
    support2 = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= 2:
        for i in range(1, len(close_1d)):
            ph = high_1d[i-1]
            pl = low_1d[i-1]
            pc = close_1d[i-1]
            
            pp = (ph + pl + pc) / 3.0
            r2 = pp + (ph - pl)
            s2 = pp - (ph - pl)
            
            pivot_point[i] = pp
            resistance2[i] = r2
            support2[i] = s2
    
    # Align 1d indicators to 6h timeframe
    pivot_point_6h = align_htf_to_ltf(prices, df_1d, pivot_point)
    resistance2_6h = align_htf_to_ltf(prices, df_1d, resistance2)
    support2_6h = align_htf_to_ltf(prices, df_1d, support2)
    
    # Volume spike detection on 6h bars
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_point_6h[i]) or 
            np.isnan(resistance2_6h[i]) or
            np.isnan(support2_6h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 6h volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above R2 with volume spike (breakout continuation)
            if (close[i] > resistance2_6h[i] and volume_ratio > 2.0):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below S2 with volume spike (breakout continuation)
            elif (close[i] < support2_6h[i] and volume_ratio > 2.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price closes below pivot (mean reversion signal)
            if close[i] < pivot_point_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price closes above pivot (mean reversion signal)
            if close[i] > pivot_point_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Pivot_S2R2_Volume_Breakout"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-14 11:39

# Strategy: 4h_Daily_Pivot_R1S1_Breakout_VolumeATR_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.023 | +20.3% | -5.5% | 168 | FAIL |
| ETHUSDT | 0.067 | +23.2% | -7.8% | 140 | PASS |
| SOLUSDT | 0.800 | +82.1% | -15.2% | 115 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.356 | +9.6% | -7.8% | 49 | PASS |
| SOLUSDT | -0.065 | +5.0% | -7.5% | 38 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Daily_Pivot_R1S1_Breakout_VolumeATR_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Daily high, low, close for pivot calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points: P = (H+L+C)/3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align daily pivot levels to 4h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Daily ATR for volatility filter (14-period)
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.absolute(high_1d[1:] - close_1d[:-1]))
    tr1 = np.maximum(tr1, np.absolute(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    atr_14_1d = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Volume confirmation: current volume > 2.5x 20-period average (4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        pivot = pivot_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        atr = atr_14_1d_aligned[i]
        
        volume_confirmed = vol > 2.5 * vol_ma
        
        if position == 0:
            # Long: break above R1 with volume
            if price > r1 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume
            elif price < s1 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price below pivot or ATR-based stop
            if price < pivot or price < close[i-1] - 2.0 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price above pivot or ATR-based stop
            if price > pivot or price > close[i-1] + 2.0 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-19 08:13

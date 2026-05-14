# Strategy: 4h_12h_Pivot_R1S1_Breakout_VolumeATR_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.393 | +36.2% | -6.5% | 318 | PASS |
| ETHUSDT | 0.789 | +60.4% | -5.4% | 293 | PASS |
| SOLUSDT | 0.468 | +52.3% | -11.9% | 230 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.954 | -8.5% | -10.8% | 116 | FAIL |
| ETHUSDT | 0.159 | +7.7% | -9.8% | 110 | PASS |
| SOLUSDT | -0.102 | +4.3% | -8.4% | 88 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Pivot_R1S1_Breakout_VolumeATR_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for pivot calculation (once before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # 12h high, low, close for pivot calculation
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h pivot points: P = (H+L+C)/3
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1_12h = 2 * pivot_12h - low_12h
    s1_12h = 2 * pivot_12h - high_12h
    
    # Align 12h pivot levels to 4h timeframe
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # 12h ATR for volatility filter (14-period)
    tr1 = np.maximum(high_12h[1:] - low_12h[1:], np.absolute(high_12h[1:] - close_12h[:-1]))
    tr1 = np.maximum(tr1, np.absolute(low_12h[1:] - close_12h[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    atr_14_12h = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    atr_14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    
    # Volume confirmation: current volume > 2.0x 20-period average (4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_12h_aligned[i]) or np.isnan(r1_12h_aligned[i]) or 
            np.isnan(s1_12h_aligned[i]) or np.isnan(atr_14_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        pivot = pivot_12h_aligned[i]
        r1 = r1_12h_aligned[i]
        s1 = s1_12h_aligned[i]
        atr = atr_14_12h_aligned[i]
        
        volume_confirmed = vol > 2.0 * vol_ma
        
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
            if price < pivot or price < close[i-1] - 1.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price above pivot or ATR-based stop
            if price > pivot or price > close[i-1] + 1.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-19 07:33

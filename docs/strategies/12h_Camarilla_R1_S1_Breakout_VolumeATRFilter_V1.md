# Strategy: 12h_Camarilla_R1_S1_Breakout_VolumeATRFilter_V1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.010 | +20.5% | -12.4% | 212 | PASS |
| ETHUSDT | 0.160 | +27.9% | -17.3% | 187 | PASS |
| SOLUSDT | -0.015 | +12.9% | -36.6% | 179 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.162 | -13.3% | -15.4% | 77 | FAIL |
| ETHUSDT | 0.391 | +11.5% | -10.1% | 72 | PASS |

## Code
```python
#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_VolumeATRFilter_V1
# Hypothesis: Camarilla pivot levels (R1/S1) from daily pivot act as institutional support/resistance.
# Breakouts with volume confirmation and ATR-based volatility filter capture sustained moves.
# Designed for 12h timeframe to reduce trade frequency and avoid fee drag. Works in both bull/bear markets
# by capturing breakouts from key levels with institutional volume validation.

name = "12h_Camarilla_R1_S1_Breakout_VolumeATRFilter_V1"
timeframe = "12h"
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
    volume = prices['volume'].values
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Using (H+L+C)/3 as pivot
    ph = df_1d['high'].values
    pl = df_1d['low'].values
    pc = df_1d['close'].values
    
    # Calculate pivot and Camarilla levels
    p = (ph + pl + pc) / 3
    r1 = p + (ph - pl) * 1.1 / 12
    s1 = p - (ph - pl) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.5)
    
    # ATR filter: only trade when volatility is sufficient
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr > (atr_ma50 * 0.5)  # Only trade when ATR > 50% of its MA
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(volume_filter[i]) or np.isnan(volatility_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + volume confirmation + sufficient volatility
            if close[i] > r1_aligned[i] and volume_filter[i] and volatility_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume confirmation + sufficient volatility
            elif close[i] < s1_aligned[i] and volume_filter[i] and volatility_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below S1 (reversal signal)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above R1 (reversal signal)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-20 01:13

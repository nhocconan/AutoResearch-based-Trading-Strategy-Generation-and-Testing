# Strategy: 6h_Camarilla_R4_S4_Breakout_With_Volume_Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.253 | +0.5% | -37.5% | 65 | FAIL |
| ETHUSDT | 0.293 | +41.9% | -33.8% | 58 | PASS |
| SOLUSDT | 1.289 | +374.7% | -43.3% | 39 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.086 | +6.2% | -15.9% | 19 | PASS |
| SOLUSDT | -0.543 | -9.3% | -31.6% | 19 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h_Camarilla_R4_S4_Breakout_With_Volume_Filter
Hypothesis: Use daily Camarilla pivot levels to identify breakout points. Go long when price breaks above S4 with volume confirmation, short when breaks below R4. Uses 1D Camarilla levels for structure and 6H volume for confirmation. Designed to work in both bull and bear markets by capturing strong momentum moves. Targets 15-25 trades/year with position size 0.25.
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
    
    # Get 1D data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R4 = Close + 1.5 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    camarilla_r4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_s4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align Camarilla levels to 6h timeframe (wait for daily bar close)
    r4_6h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate volume average (20-period) for confirmation
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: price breaks above S4 with volume confirmation
            if close[i] > s4_6h[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below R4 with volume confirmation
            elif close[i] < r4_6h[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses back below S4
            if close[i] < s4_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above R4
            if close[i] > r4_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R4_S4_Breakout_With_Volume_Filter"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-18 10:58

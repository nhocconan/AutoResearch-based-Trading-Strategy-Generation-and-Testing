# Strategy: 4h_Pivot_R1S1_Breakout_12hEMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.108 | +25.0% | -11.4% | 123 | PASS |
| ETHUSDT | 0.233 | +34.1% | -12.1% | 119 | PASS |
| SOLUSDT | 0.857 | +145.8% | -24.3% | 98 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.300 | -8.0% | -12.8% | 51 | FAIL |
| ETHUSDT | 0.505 | +14.8% | -10.0% | 44 | PASS |
| SOLUSDT | 0.227 | +9.3% | -16.8% | 32 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Pivot_R1S1_Breakout_12hEMA34_Volume
Hypothesis: Daily Camarilla R1/S1 breakout with 12-hour EMA34 trend filter and volume confirmation.
Trades breakouts from key daily pivot levels only when aligned with 12-hour trend and accompanied by volume spike,
avoiding false breakouts in low-volume or counter-trend conditions. Designed for low frequency (~20-40 trades/year)
with strong performance across bull and bear markets by combining daily structure with intermediate trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA34 trend filter
    close_12h = df_12h['close'].values
    ema34_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 34:
        ema34_12h[33] = np.mean(close_12h[0:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_12h)):
            ema34_12h[i] = close_12h[i] * alpha + ema34_12h[i-1] * (1 - alpha)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels (H3/L3 from prior day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    r1 = np.full(len(close_1d), np.nan)  # H3 level
    s1 = np.full(len(close_1d), np.nan)  # L3 level
    
    for i in range(1, len(close_1d)):
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        diff = ph - pl
        r1[i] = pc + 1.0 * diff  # H3
        s1[i] = pc - 1.0 * diff  # L3
    
    # Align 12h EMA and daily levels to 4h timeframe
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above daily R1 with volume spike and 12h uptrend
            if (close[i] > r1_aligned[i] and vol_spike[i] and 
                close[i] > ema34_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below daily S1 with volume spike and 12h downtrend
            elif (close[i] < s1_aligned[i] and vol_spike[i] and 
                  close[i] < ema34_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below daily S1 or 12h trend turns down
            if (close[i] < s1_aligned[i] or close[i] < ema34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above daily R1 or 12h trend turns up
            if (close[i] > r1_aligned[i] or close[i] > ema34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_R1S1_Breakout_12hEMA34_Volume"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-18 09:38

# Strategy: 4h_Donchian_Breakout_20_Trend_Plus_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.205 | +30.8% | -16.8% | 108 | PASS |
| ETHUSDT | 0.116 | +25.2% | -18.0% | 113 | PASS |
| SOLUSDT | 0.726 | +116.3% | -24.6% | 108 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.375 | -8.5% | -10.6% | 50 | FAIL |
| ETHUSDT | 0.136 | +7.5% | -10.1% | 38 | PASS |
| SOLUSDT | 0.570 | +16.8% | -12.1% | 33 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Donchian_Breakout_20_Trend_Plus_Volume
Hypothesis: 4-hour Donchian breakouts above upper channel or below lower channel with daily EMA34 trend filter and volume confirmation.
Donchian channels provide clear breakout levels, daily EMA34 filters trend direction, volume confirms breakout strength.
Designed for low trade frequency (target: 20-50/year) with strong performance in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 with proper smoothing
    ema34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[0:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34_1d[i] = close_1d[i] * alpha + ema34_1d[i-1] * (1 - alpha)
    
    # Align daily EMA34 to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Donchian channels (20-period high/low)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    # Volume spike: current volume > 1.8 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above Donchian upper with volume spike and daily uptrend
            if (close[i] > upper[i] and vol_spike[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian lower with volume spike and daily downtrend
            elif (close[i] < lower[i] and vol_spike[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below Donchian lower or daily trend turns down
            if (close[i] < lower[i] or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above Donchian upper or daily trend turns up
            if (close[i] > upper[i] or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_20_Trend_Plus_Volume"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-18 09:14

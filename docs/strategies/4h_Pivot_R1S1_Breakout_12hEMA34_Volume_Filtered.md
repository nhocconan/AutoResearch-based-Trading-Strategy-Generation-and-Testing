# Strategy: 4h_Pivot_R1S1_Breakout_12hEMA34_Volume_Filtered

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.566 | +43.8% | -6.4% | 441 | PASS |
| ETHUSDT | 0.692 | +55.5% | -7.7% | 405 | PASS |
| SOLUSDT | 0.777 | +85.0% | -18.9% | 340 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.820 | -0.5% | -4.6% | 167 | FAIL |
| ETHUSDT | 1.492 | +26.3% | -6.3% | 148 | PASS |
| SOLUSDT | 0.600 | +13.6% | -6.4% | 122 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Pivot_R1S1_Breakout_12hEMA34_Volume_Filtered
Hypothesis: Combines 1d Camarilla R1/S1 breakout with 12h EMA34 trend filter and volume confirmation to reduce false breakouts. Uses tighter exit and minimum hold to control trade frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 12h data for EMA34 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r1 = np.zeros_like(close_1d)
    camarilla_s1 = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i == 0:
            camarilla_r1[i] = close_1d[i]
            camarilla_s1[i] = close_1d[i]
        else:
            rang = high_1d[i-1] - low_1d[i-1]
            camarilla_r1[i] = close_1d[i-1] + rang * 1.1 / 12
            camarilla_s1[i] = close_1d[i-1] - rang * 1.1 / 12
    
    # Align 1d Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = np.zeros_like(close_12h)
    ema_34_12h[:] = np.nan
    if len(close_12h) >= 34:
        k = 2 / (34 + 1)
        ema_34_12h[33] = np.mean(close_12h[:34])
        for i in range(34, len(close_12h)):
            ema_34_12h[i] = close_12h[i] * k + ema_34_12h[i-1] * (1 - k)
    
    # Align 12h EMA34 to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-20+1:i+1])
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = 35  # Warmup for EMA
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and above 12h EMA34
            if close[i] > camarilla_r1_aligned[i] and vol_spike[i] and close[i] > ema_34_aligned[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: price breaks below S1 with volume spike and below 12h EMA34
            elif close[i] < camarilla_s1_aligned[i] and vol_spike[i] and close[i] < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Exit: minimum 4 bars hold, then exit on mean reversion or trend change
            if bars_since_entry >= 4:
                if close[i] < camarilla_s1_aligned[i] or close[i] < ema_34_aligned[i] or not vol_spike[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25  # Hold during minimum period
        
        elif position == -1:
            # Exit: minimum 4 bars hold, then exit on mean reversion or trend change
            if bars_since_entry >= 4:
                if close[i] > camarilla_r1_aligned[i] or close[i] > ema_34_aligned[i] or not vol_spike[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25  # Hold during minimum period
    
    return signals

name = "4h_Pivot_R1S1_Breakout_12hEMA34_Volume_Filtered"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-18 07:54

# Strategy: 12h_1d_Camarilla_Pivot_Breakout_Volume_Trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.632 | -0.3% | -16.4% | 195 | FAIL |
| ETHUSDT | 0.083 | +23.8% | -17.0% | 160 | PASS |
| SOLUSDT | 0.318 | +42.7% | -20.6% | 152 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.330 | +9.8% | -5.8% | 60 | PASS |
| SOLUSDT | -0.396 | +0.5% | -10.1% | 55 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
12h_1d_Camarilla_Pivot_Breakout_Volume_Trend_v1
Hypothesis: 12h timeframe with 1d Camarilla pivot levels, volume confirmation, and 1d EMA trend filter.
Designed for fewer trades (target 12-37/year) by requiring breakouts of H3/L3 levels with volume > 1.5x average
and price aligned with 1d EMA trend. Works in bull/bear markets by only taking trend-aligned breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_Pivot_Breakout_Volume_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivots and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate pivot and ranges
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    H3 = pivot + range_hl * 1.1 / 4
    L3 = pivot - range_hl * 1.1 / 4
    H4 = pivot + range_hl * 1.1 / 2
    L4 = pivot - range_hl * 1.1 / 2
    
    # Align to 12h timeframe
    H3_12h = align_htf_to_ltf(prices, df_1d, H3)
    L3_12h = align_htf_to_ltf(prices, df_1d, L3)
    H4_12h = align_htf_to_ltf(prices, df_1d, H4)
    L4_12h = align_htf_to_ltf(prices, df_1d, L4)
    
    # Calculate 1d EMA (21 period) for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume average (20 period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(H3_12h[i]) or np.isnan(L3_12h[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume spike: current volume > 1.5x average
        volume_spike = volume[i] > vol_ma[i] * 1.5
        
        # Trend filter: price above/below 1d EMA
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        # Entry conditions: breakout of H3/L3 with volume and trend
        long_entry = (close[i] > H3_12h[i]) and volume_spike and above_ema
        short_entry = (close[i] < L3_12h[i]) and volume_spike and below_ema
        
        # Exit conditions: return to H4/L4 levels or trend reversal
        long_exit = (close[i] < H4_12h[i]) or (close[i] < ema_1d_aligned[i])
        short_exit = (close[i] > L4_12h[i]) or (close[i] > ema_1d_aligned[i])
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
```

## Last Updated
2026-04-12 00:19

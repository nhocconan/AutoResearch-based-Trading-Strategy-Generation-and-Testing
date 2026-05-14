# Strategy: 12h_1d_Camarilla_Pivot_R1S1_Breakout_Volume_Trend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.686 | +2.4% | -11.2% | 102 | FAIL |
| ETHUSDT | 0.032 | +21.8% | -10.5% | 84 | PASS |
| SOLUSDT | 0.032 | +19.4% | -21.0% | 78 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.545 | +11.7% | -4.1% | 36 | PASS |
| SOLUSDT | -0.153 | +4.0% | -11.4% | 32 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
12h_1d_Camarilla_Pivot_R1S1_Breakout_Volume_Trend
Hypothesis: Breakout of 1d R1/S1 levels with volume confirmation and 12h trend bias.
Trades only in the direction of the 12h EMA trend to avoid whipsaws in choppy markets.
Targets 12-37 trades per year by using strict daily pivot levels, volume confirmation, and trend filter.
Works in both bull and bear markets by following the 12h trend.
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
    
    # Get 1d data for pivot levels (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    rng_1d = high_1d - low_1d
    r1_1d = close_1d + rng_1d * 1.1 / 12
    s1_1d = close_1d - rng_1d * 1.1 / 12
    
    # Calculate 1d pivot for trend bias
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    
    # Align all levels to 12h timeframe (wait for bar close)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Get 12h trend (EMA34) for directional bias
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(pivot_1d_aligned[i]) or np.isnan(ema_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above 1d R1, above 1d pivot, with volume, and 12h uptrend
            if (close[i] > r1_1d_aligned[i] and 
                close[i] > pivot_1d_aligned[i] and vol_confirm[i] and 
                close[i] > ema_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below 1d S1, below 1d pivot, with volume, and 12h downtrend
            elif (close[i] < s1_1d_aligned[i] and 
                  close[i] < pivot_1d_aligned[i] and vol_confirm[i] and 
                  close[i] < ema_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns to 1d S1 or 12h downtrend
            if (not np.isnan(s1_1d_aligned[i]) and close[i] < s1_1d_aligned[i]) or \
               (close[i] < ema_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to 1d R1 or 12h uptrend
            if (not np.isnan(r1_1d_aligned[i]) and close[i] > r1_1d_aligned[i]) or \
               (close[i] > ema_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Camarilla_Pivot_R1S1_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-18 10:18

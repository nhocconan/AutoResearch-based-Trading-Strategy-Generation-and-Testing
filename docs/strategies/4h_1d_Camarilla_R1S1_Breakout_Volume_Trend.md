# Strategy: 4h_1d_Camarilla_R1S1_Breakout_Volume_Trend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.147 | +14.0% | -13.8% | 351 | FAIL |
| ETHUSDT | 0.242 | +32.5% | -11.1% | 329 | PASS |
| SOLUSDT | 0.453 | +56.3% | -26.6% | 280 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.926 | +19.3% | -5.8% | 125 | PASS |
| SOLUSDT | 0.401 | +11.3% | -11.7% | 102 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_1d_Camarilla_R1S1_Breakout_Volume_Trend
Hypothesis: Breakout of daily R1/S1 levels with volume confirmation and 4h EMA trend bias.
Trades in direction of 4h EMA to avoid whipsaws. Targets 20-50 trades/year by using strict
daily pivot breakouts, volume > 2x 20-period average, and EMA34 filter.
Works in bull/bear markets by following trend.
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
    
    # Get 1d data for Camarilla levels (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla R1 and S1
    rng_1d = high_1d - low_1d
    r1_1d = close_1d + rng_1d * 1.1 / 12
    s1_1d = close_1d - rng_1d * 1.1 / 12
    
    # Calculate 1d pivot for trend bias
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    
    # Align all levels to 4h timeframe (wait for bar close)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Get 4h trend (EMA34) for directional bias
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
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
            np.isnan(pivot_1d_aligned[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above 1d R1, above 1d pivot, with volume, and 4h uptrend
            if (close[i] > r1_1d_aligned[i] and 
                close[i] > pivot_1d_aligned[i] and vol_confirm[i] and 
                close[i] > ema_4h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below 1d S1, below 1d pivot, with volume, and 4h downtrend
            elif (close[i] < s1_1d_aligned[i] and 
                  close[i] < pivot_1d_aligned[i] and vol_confirm[i] and 
                  close[i] < ema_4h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns to 1d S1 or 4h downtrend
            if (not np.isnan(s1_1d_aligned[i]) and close[i] < s1_1d_aligned[i]) or \
               (close[i] < ema_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to 1d R1 or 4h uptrend
            if (not np.isnan(r1_1d_aligned[i]) and close[i] > r1_1d_aligned[i]) or \
               (close[i] > ema_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_Camarilla_R1S1_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-18 10:20

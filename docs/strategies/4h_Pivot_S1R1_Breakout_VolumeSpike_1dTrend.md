# Strategy: 4h_Pivot_S1R1_Breakout_VolumeSpike_1dTrend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.144 | +26.9% | -11.8% | 156 | PASS |
| ETHUSDT | 0.041 | +20.5% | -13.8% | 149 | PASS |
| SOLUSDT | 0.963 | +154.8% | -22.4% | 127 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.077 | -5.2% | -7.8% | 59 | FAIL |
| ETHUSDT | 0.790 | +19.6% | -10.1% | 49 | PASS |
| SOLUSDT | 0.031 | +5.6% | -12.5% | 42 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Pivot_S1R1_Breakout_VolumeSpike_1dTrend
Hypothesis: Daily pivot S1 and R1 levels act as strong support/resistance. 
Breakouts beyond these levels with volume confirmation and daily EMA trend filter capture momentum.
Designed for 20-30 trades/year on 4h timeframe with low trade frequency to minimize fee flood.
Works in bull/bear markets by requiring volume spike and daily EMA trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points using standard formula
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d = df_1d['close']
    
    pivot = (high_1d + low_1d + close_1d) / 3
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Shift by 1 to use previous day's levels only
    r1_prev = r1.shift(1).values
    s1_prev = s1.shift(1).values
    
    # Align to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_prev)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_prev)
    
    # Get daily data for EMA trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: 2x 20-period average on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: break above R1 with volume spike and price above daily EMA (uptrend)
            if price > r1_val and volume_spike[i] and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume spike and price below daily EMA (downtrend)
            elif price < s1_val and volume_spike[i] and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price returns to S1 or breaks below daily EMA
            if price <= s1_val or price < ema_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price returns to R1 or breaks above daily EMA
            if price >= r1_val or price > ema_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Pivot_S1R1_Breakout_VolumeSpike_1dTrend"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-18 01:45

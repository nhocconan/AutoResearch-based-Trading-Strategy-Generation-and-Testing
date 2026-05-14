# Strategy: 4h_12h_1d_camarilla_breakout_volume_v5

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.146 | -18.8% | -27.4% | 365 | FAIL |
| ETHUSDT | -0.233 | +7.4% | -20.3% | 361 | FAIL |
| SOLUSDT | 0.161 | +28.5% | -24.5% | 346 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.407 | +11.8% | -8.6% | 118 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_12h_1d_camarilla_breakout_volume_v5
Hypothesis: Combine 12h trend (EMA25) with 1d Camarilla levels (R1/S1) and 4h breakout confirmation.
Trade only when 4h price breaks 1d R1/S1 in direction of 12h trend with volume confirmation.
Exit when price returns to 1d pivot or 12h trend reverses.
Designed for ~20-30 trades/year to avoid fee drag, works in bull/bear via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_1d_camarilla_breakout_volume_v5"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h EMA(25) for trend
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and ranges
    pivot = (high_1d + low_1d + close_1d) / 3
    range_ = high_1d - low_1d
    
    # Key Camarilla levels: R1 and S1 (most significant for intraday)
    r1 = close_1d + range_ * 1.1 / 12
    s1 = close_1d - range_ * 1.1 / 12
    pivot_level = pivot  # for exit
    
    # Align to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_level)
    
    # Volume confirmation: volume > 1.3x average of last 20 periods (20*4h = ~3.3 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.3
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to pivot or trend changes (12h EMA turns down)
            if close[i] <= pivot_aligned[i] or ema_12h_aligned[i] < ema_12h_aligned[max(0, i-1)]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price returns to pivot or trend changes (12h EMA turns up)
            if close[i] >= pivot_aligned[i] or ema_12h_aligned[i] > ema_12h_aligned[max(0, i-1)]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above R1 with volume and 12h uptrend
            if (close[i] > r1_aligned[i] and 
                ema_12h_aligned[i] > ema_12h_aligned[max(0, i-1)] and  # Uptrend confirmation
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below S1 with volume and 12h downtrend
            elif (close[i] < s1_aligned[i] and 
                  ema_12h_aligned[i] < ema_12h_aligned[max(0, i-1)] and  # Downtrend confirmation
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 10:44

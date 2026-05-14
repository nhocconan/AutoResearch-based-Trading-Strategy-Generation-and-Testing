# Strategy: 12H_Camarilla_R1_S1_Breakout_1D_Trend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.295 | +8.8% | -10.7% | 24 | FAIL |
| ETHUSDT | 0.087 | +23.7% | -12.5% | 22 | PASS |
| SOLUSDT | 0.567 | +73.0% | -22.9% | 21 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.695 | +17.7% | -7.6% | 6 | PASS |
| SOLUSDT | -0.183 | +1.2% | -14.8% | 8 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
12H_Camarilla_R1_S1_Breakout_1D_Trend_Volume
Hypothesis: 12h price breaks above/below 1D Camarilla R1/S1 levels with 1D EMA34 trend confirmation and volume spike.
Works in bull/bear markets: R1/S1 breakouts capture strong moves while avoiding minor retracements.
EMA34 filter ensures alignment with daily trend, volume confirmation validates breakout strength.
Targets 12-37 trades/year to minimize fee drag on 12h timeframe.
"""
name = "12H_Camarilla_R1_S1_Breakout_1D_Trend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 1D data for Camarilla levels, EMA trend, and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1D Camarilla levels (R1, S1)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1 = pivot + (range_1d * 1.1 / 6)  # R1 level
    s1 = pivot - (range_1d * 1.1 / 6)  # S1 level
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 1D EMA34 for trend direction
    close_1d_series = pd.Series(df_1d['close'])
    ema_34 = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume filter: current 12h volume > 1.5 x 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(34, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 48 bars between trades (8 days on 12h TF) to reduce frequency
            if bars_since_exit < 48:
                continue
                
            # Long: price breaks above R1 with EMA34 uptrend and volume spike
            if (close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1] and 
                close[i] > ema_34_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price breaks below S1 with EMA34 downtrend and volume spike
            elif (close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1] and 
                  close[i] < ema_34_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite EMA34 side (trend reversal)
            if position == 1 and close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals
```

## Last Updated
2026-05-07 08:10

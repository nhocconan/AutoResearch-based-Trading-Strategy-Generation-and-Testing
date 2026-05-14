# Strategy: 6H_Camarilla_R1_S1_Breakout_1dEMA34_1wTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.086 | +23.9% | -11.3% | 79 | KEEP |
| ETHUSDT | 0.125 | +25.9% | -10.7% | 69 | KEEP |
| SOLUSDT | 0.874 | +120.0% | -23.2% | 70 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.725 | +1.9% | -5.5% | 23 | DISCARD |
| ETHUSDT | 0.281 | +8.9% | -5.5% | 18 | KEEP |
| SOLUSDT | 1.021 | +16.0% | -4.9% | 12 | KEEP |

## Code
```python
# 6H_Camarilla_R1_S1_Breakout_1dEMA34_1wTrend_Volume
# Hypothesis: Combines Camarilla pivot breakouts with multi-timeframe trend filtering and volume confirmation.
# Uses daily EMA34 for intermediate trend and weekly pivot levels for higher timeframe bias.
# Volume spikes confirm breakout strength. Designed to work in both bull and bear markets by requiring
# alignment across timeframes, reducing false signals during choppy periods.
# Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.

#!/usr/bin/env python3
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
    
    # Load 1d and 1w data for pivot points and EMA (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 35 or len(df_1w) < 10:  # Need enough for EMA34
        return np.zeros(n)
    
    # Previous day's pivot points (standard)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels
    pivot = (high_1d + low_1d + close_1d) / 3
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Previous week's pivot points (for trend filter)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot levels
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34 = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema_34_aligned[i]) or
            np.isnan(vol_avg_20[i]) or np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume spike AND above 1d EMA34 (uptrend) AND above weekly R1 (strong uptrend)
            if (close[i] > r1_aligned[i] and volume[i] > 1.8 * vol_avg_20[i] and 
                close[i] > ema_34_aligned[i] and close[i] > r1_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume spike AND below 1d EMA34 (downtrend) AND below weekly S1 (strong downtrend)
            elif (close[i] < s1_aligned[i] and volume[i] > 1.8 * vol_avg_20[i] and 
                  close[i] < ema_34_aligned[i] and close[i] < s1_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back to opposite level (S1 for long, R1 for short)
            if position == 1:
                # Exit long: Price closes below S1
                if close[i] < s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Price closes above R1
                if close[i] > r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_Camarilla_R1_S1_Breakout_1dEMA34_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-22 12:18

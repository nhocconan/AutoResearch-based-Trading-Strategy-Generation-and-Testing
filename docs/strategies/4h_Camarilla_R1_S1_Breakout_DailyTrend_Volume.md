# Strategy: 4h_Camarilla_R1_S1_Breakout_DailyTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.476 | +38.6% | -5.6% | 264 | PASS |
| ETHUSDT | 0.255 | +31.8% | -9.8% | 243 | PASS |
| SOLUSDT | 0.683 | +77.4% | -16.8% | 206 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.608 | -5.7% | -7.5% | 102 | FAIL |
| ETHUSDT | 1.334 | +23.7% | -5.1% | 89 | PASS |
| SOLUSDT | 0.921 | +17.7% | -6.1% | 71 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h Camarilla Pivot R1/S1 Breakout with Volume Spike and Daily Trend Filter
Hypothesis: Camarilla pivot levels (R1/S1) act as intraday support/resistance.
Breaks with volume confirmation and daily EMA trend filter capture breakouts
in both bull and bear markets while avoiding false signals. Designed for low
trade frequency (target: 20-50 trades/year) to minimize fee drag.
"""
name = "4h_Camarilla_R1_S1_Breakout_DailyTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY TREND (EMA 34) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === DAILY CAMARILLA PIVOT LEVELS ===
    # Typical price for pivot calculation
    typical_price = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    # Pivot point
    pivot = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    # R1 and S1 levels
    r1 = pivot + (1.1/12) * (df_1d['high'].values - df_1d['low'].values)
    s1 = pivot - (1.1/12) * (df_1d['high'].values - df_1d['low'].values)
    # Align to 4h timeframe
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 4H VOLUME (20) SPIKE ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trend_1d[i]) or np.isnan(pivot_4h[i]) or np.isnan(r1_4h[i]) or 
            np.isnan(s1_4h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Above daily EMA + break above R1 + volume spike
            if (close[i] > trend_1d[i] and 
                close[i] > r1_4h[i] and
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Below daily EMA + break below S1 + volume spike
            elif (close[i] < trend_1d[i] and 
                  close[i] < s1_4h[i] and
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Close below pivot OR volume dries up
            if close[i] < pivot_4h[i] or volume[i] < vol_ma[i] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above pivot OR volume dries up
            if close[i] > pivot_4h[i] or volume[i] < vol_ma[i] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 06:28

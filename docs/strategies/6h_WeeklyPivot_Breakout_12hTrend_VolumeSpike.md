# Strategy: 6h_WeeklyPivot_Breakout_12hTrend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.177 | +29.6% | -13.2% | 113 | PASS |
| ETHUSDT | 0.528 | +64.8% | -12.0% | 97 | PASS |
| SOLUSDT | 1.151 | +263.9% | -33.6% | 104 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.293 | -3.3% | -6.3% | 29 | FAIL |
| ETHUSDT | 0.122 | +7.2% | -8.4% | 23 | PASS |
| SOLUSDT | -0.331 | -0.2% | -12.4% | 24 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 6h_WeeklyPivot_Breakout_12hTrend_VolumeSpike
# Hypothesis: On 6h timeframe, enter long when price closes above weekly S2 with close > 12h EMA50 and volume spike.
# Enter short when price closes below weekly R2 with close < 12h EMA50 and volume spike.
# Exit when price crosses 12h EMA50 (trend reversal).
# Uses weekly timeframe for pivot levels and 12h for trend filter to avoid short-term noise.
# Targets 15-30 trades/year for low fee drag and works in both bull and bear markets by fading extreme weekly levels.

name = "6h_WeeklyPivot_Breakout_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate weekly pivot point and range
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_range = weekly_high - weekly_low
    
    # Weekly R2 and S2 levels (stronger levels for breakout)
    r2 = weekly_pivot + weekly_range * 1.1000 / 4.0
    s2 = weekly_pivot - weekly_range * 1.1000 / 4.0
    
    # Load 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate 12h EMA50
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 6h timeframe
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        ema12h_trend = ema50_12h_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # LONG: Price closes above S2 with close > 12h EMA50 and volume > 1.5x MA
            if close[i] > s2_val and close[i] > ema12h_trend and volume[i] > vol_ma_val * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below R2 with close < 12h EMA50 and volume > 1.5x MA
            elif close[i] < r2_val and close[i] < ema12h_trend and volume[i] > vol_ma_val * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 12h EMA50 (trend reversal)
            if close[i] < ema12h_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above 12h EMA50 (trend reversal)
            if close[i] > ema12h_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 08:59

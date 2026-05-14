# Strategy: 6h_DailyPivot_Breakout_EMA50_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.930 | -14.1% | -32.3% | 345 | FAIL |
| ETHUSDT | 0.009 | +19.4% | -16.1% | 326 | PASS |
| SOLUSDT | 1.221 | +179.3% | -14.4% | 273 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.714 | +16.2% | -8.8% | 104 | PASS |
| SOLUSDT | 0.399 | +11.6% | -8.9% | 95 | PASS |

## Code
```python
# 6h_DailyPivot_Breakout_EMA50_Volume
# Hypothesis: 6-hour breakout above/below daily pivot levels with EMA50 trend filter and volume confirmation.
# Works in bull markets (breakouts above pivot + above EMA50) and bear markets (breakdowns below pivot + below EMA50).
# Uses daily pivot levels (calculated from prior day's OHLC) as support/resistance.
# EMA50 filter ensures we trade in direction of intermediate trend.
# Volume confirmation adds conviction to breakouts.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

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
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # R2 = P + (H - L)
    # S2 = P - (H - L)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    r2_1d = pivot_1d + (high_1d - low_1d)
    s2_1d = pivot_1d - (high_1d - low_1d)
    
    # Align daily pivot levels to 6h timeframe (use previous day's levels)
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # Calculate 6h EMA50 for trend filter
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current volume > 1.3 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need sufficient data for EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(ema50[i]) or np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.3 * volume_ma20[i])
        
        # Trend filter: price relative to EMA50
        price_above_ema = close[i] > ema50[i]
        price_below_ema = close[i] < ema50[i]
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume and above EMA50
            if (close[i] > r1_6h[i] and volume_filter and price_above_ema):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with volume and below EMA50
            elif (close[i] < s1_6h[i] and volume_filter and price_below_ema):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below pivot or EMA50
            if close[i] < pivot_6h[i] or close[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above pivot or EMA50
            if close[i] > pivot_6h[i] or close[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_DailyPivot_Breakout_EMA50_Volume"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-17 11:53

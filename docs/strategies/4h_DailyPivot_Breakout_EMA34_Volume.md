# Strategy: 4h_DailyPivot_Breakout_EMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.394 | +3.2% | -18.1% | 402 | FAIL |
| ETHUSDT | 0.026 | +20.2% | -13.5% | 378 | PASS |
| SOLUSDT | 0.363 | +49.7% | -26.1% | 347 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.050 | +6.2% | -11.1% | 129 | PASS |
| SOLUSDT | 0.015 | +5.5% | -13.6% | 124 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align daily pivot levels to 4h timeframe (use previous day's levels)
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate 4h EMA34 for trend filter
    close_series = pd.Series(close)
    ema34 = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 34  # Need sufficient data for EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_4h[i]) or np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or
            np.isnan(ema34[i]) or np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price relative to EMA34
        price_above_ema = close[i] > ema34[i]
        price_below_ema = close[i] < ema34[i]
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume and above EMA34
            if (close[i] > r1_4h[i] and volume_filter and price_above_ema):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with volume and below EMA34
            elif (close[i] < s1_4h[i] and volume_filter and price_below_ema):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below pivot or EMA34
            if close[i] < pivot_4h[i] or close[i] < ema34[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above pivot or EMA34
            if close[i] > pivot_4h[i] or close[i] > ema34[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DailyPivot_Breakout_EMA34_Volume"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-17 12:04

# Strategy: 4h_Fibonacci_Retracement_Breakout_Volume_12hTrend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.141 | +14.6% | -11.6% | 469 | FAIL |
| ETHUSDT | 0.385 | +40.9% | -11.0% | 435 | PASS |
| SOLUSDT | 0.577 | +70.6% | -23.8% | 405 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.801 | +17.5% | -6.3% | 145 | PASS |
| SOLUSDT | 1.220 | +25.2% | -7.2% | 131 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h Fibonacci Retracement Breakout with Volume Confirmation and 12h Trend Filter.
Long when price breaks above 61.8% retracement level AND price > 12h EMA50 AND volume > 1.5x average.
Short when price breaks below 38.2% retracement level AND price < 12h EMA50 AND volume > 1.5x average.
Exit when price crosses back through 50% retracement level.
Uses 12h swing high/low calculated from prior 12h period for zero look-ahead.
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
    
    # Get 12h data for swing points and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h swing high and low (using prior 12h period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h range
    range_12h = high_12h - low_12h
    
    # Fibonacci levels (based on prior 12h swing)
    level_618 = low_12h + 0.618 * range_12h  # 61.8% retracement
    level_500 = low_12h + 0.500 * range_12h  # 50% retracement
    level_382 = low_12h + 0.382 * range_12h  # 38.2% retracement
    
    # Align 12h levels to 4h timeframe (use prior 12h period's levels)
    level_618_aligned = align_htf_to_ltf(prices, df_12h, level_618)
    level_500_aligned = align_htf_to_ltf(prices, df_12h, level_500)
    level_382_aligned = align_htf_to_ltf(prices, df_12h, level_382)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter: volume > 1.5x 24-period average (6 hours)
    vol_ma_24 = np.full(n, np.nan, dtype=np.float64)
    for i in range(23, n):
        vol_ma_24[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 12h data (1 period) + volume MA
    start_idx = max(23, 50)  # Need EMA50 and vol MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(level_618_aligned[i]) or np.isnan(level_500_aligned[i]) or 
            np.isnan(level_382_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price_now = close[i]
        level_618_val = level_618_aligned[i]
        level_500_val = level_500_aligned[i]
        level_382_val = level_382_aligned[i]
        ema_trend = ema_50_12h_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_24[i]
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: price breaks above 61.8% level AND price > 12h EMA50 AND volume spike
            if price_now > level_618_val and price_now > ema_trend and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below 38.2% level AND price < 12h EMA50 AND volume spike
            elif price_now < level_382_val and price_now < ema_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 50% level
            if price_now < level_500_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above 50% level
            if price_now > level_500_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Fibonacci_Retracement_Breakout_Volume_12hTrend"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-27 09:02

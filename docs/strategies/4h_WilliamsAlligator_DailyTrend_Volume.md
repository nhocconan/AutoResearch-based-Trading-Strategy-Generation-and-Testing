# Strategy: 4h_WilliamsAlligator_DailyTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.103 | +12.0% | -17.3% | 146 | FAIL |
| ETHUSDT | 0.317 | +42.5% | -14.4% | 138 | PASS |
| SOLUSDT | 1.189 | +271.7% | -23.3% | 118 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.264 | +10.1% | -9.8% | 56 | PASS |
| SOLUSDT | -0.098 | +2.2% | -17.1% | 41 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h Williams Alligator with 1-day Trend Filter and Volume Spike.
Long when price above Alligator's Jaw (teeth) + daily trend up + volume spike.
Short when price below Alligator's Jaw + daily trend down + volume spike.
Exit when price crosses back below/above Jaw or trend changes.
Designed for low frequency (15-30 trades/year) to minimize fee drag.
Uses Williams Alligator (SMMA: 13,8,5) as trend/filter system.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (used in Williams Alligator)"""
    if length < 1:
        return source
    result = np.full_like(source, np.nan, dtype=np.float64)
    if len(source) < length:
        return result
    # First value is simple average
    result[length-1] = np.mean(source[:length])
    # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CURRENT) / length
    for i in range(length, len(source)):
        result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator components on daily timeframe
    # Jaw (Blue): 13-period SMMA of median price, shifted 8 bars forward
    # Teeth (Red): 8-period SMMA of median price, shifted 5 bars forward  
    # Lips (Green): 5-period SMMA of median price, shifted 3 bars forward
    median_price = (df_1d['high'].values + df_1d['low'].values) / 2
    
    jaw_raw = smma(median_price, 13)
    teeth_raw = smma(median_price, 8)
    lips_raw = smma(median_price, 5)
    
    # Align to lower timeframe (4h) with proper delay
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_raw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_raw)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_raw)
    
    # For trading signals, we use the Teeth (8-period) as the main trend indicator
    # Jaw acts as support/resistance in trending markets
    trend_indicator = teeth_aligned  # Primary trend filter
    support_level = jaw_aligned      # Dynamic support/resistance
    
    # Volume filter: volume > 2.0x average (to avoid false breakouts)
    vol_ma_20 = np.empty_like(volume, dtype=np.float64)
    vol_ma_20.fill(np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Alligator components (max 13 periods) + volume MA (20)
    start_idx = max(13, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trend_indicator[i]) or np.isnan(support_level[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        trend = trend_indicator[i]
        support = support_level[i]
        
        # Volume filter: volume > 2.0x average
        vol_filter = vol_now > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Bull: price above support (Jaw) + price above trend (Teeth) + volume spike
            if price_now > support and price_now > trend and vol_filter:
                signals[i] = size
                position = 1
            # Bear: price below support (Jaw) + price below trend (Teeth) + volume spike
            elif price_now < support and price_now < trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back below support (Jaw) or trend turns down
            if price_now < support or price_now < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses back above support (Jaw) or trend turns up
            if price_now > support or price_now > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_WilliamsAlligator_DailyTrend_Volume"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-27 08:30

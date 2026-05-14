# Strategy: 4h_1d_donchian_volume_v3

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.305 | +15.1% | -7.1% | 214 | FAIL |
| ETHUSDT | -0.643 | +4.3% | -7.4% | 217 | FAIL |
| SOLUSDT | 0.181 | +29.1% | -18.4% | 200 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.621 | +11.3% | -3.3% | 73 | PASS |

## Code
```python
#!/usr/bin/env python3
# [24901] 4h_1d_donchian_volume_v3
# Hypothesis: 4-hour Donchian(20) breakout with volume confirmation and 1-day trend filter.
# Long when price breaks above 20-period Donchian high with volume > 1.8x average and price > 1-day EMA50.
# Short when price breaks below 20-period Donchian low with volume > 1.8x average and price < 1-day EMA50.
# Exit when price crosses the opposite Donchian boundary or volume falls below 1.3x average.
# Uses tighter entry conditions (volume > 1.8x) to limit trades (~15-25/year) and reduce fee drag.
# Designed to work in both bull and bear markets by combining breakout momentum with trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = np.full_like(close_1d, np.nan, dtype=float)
    if len(close_1d) >= 50:
        alpha = 2.0 / (50 + 1)
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_50_1d[i-1]
    
    # Calculate 4-hour Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 1-day EMA50 to 4-hour timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        dh = donchian_high[i]
        dl = donchian_low[i]
        trend_up_1d = price > ema_50_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: price crosses below Donchian low or volume drops below 1.3x average
            if price < dl or vol_ratio < 1.3:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above Donchian high or volume drops below 1.3x average
            if price > dh or vol_ratio < 1.3:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high with volume expansion and uptrend on 1d
            if price > dh and vol_ratio > 1.8 and trend_up_1d:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low with volume expansion and downtrend on 1d
            elif price < dl and vol_ratio > 1.8 and not trend_up_1d:
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 21:43

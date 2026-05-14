# Strategy: 6h_1d_Donchian_Pivot_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.484 | +1.3% | -22.4% | 174 | FAIL |
| ETHUSDT | 0.436 | +45.5% | -8.1% | 161 | PASS |
| SOLUSDT | 0.498 | +64.9% | -22.0% | 155 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.573 | +14.5% | -6.9% | 54 | PASS |
| SOLUSDT | 0.135 | +7.4% | -11.1% | 55 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction and volume confirmation.
# Long: price breaks above Donchian(20) high + price above 1d weekly pivot + volume > 1.5x avg volume
# Short: price breaks below Donchian(20) low + price below 1d weekly pivot + volume > 1.5x avg volume
# Weekly pivot calculated from 1d data: PP = (high+low+close)/3, R1 = 2*PP - low, S1 = 2*PP - high
# Trend filter: only take longs when price > PP, shorts when price < PP
# Volume confirmation reduces false breakouts
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Works in both bull and bear markets by using 1d pivot as trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for weekly pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly pivot point (using prior day's data)
    pp = np.full(len(close_1d), np.nan)
    for i in range(1, len(close_1d)):
        pp[i] = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3.0
    
    # Align 1d pivot to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Donchian(20) on 6h timeframe
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Average volume (20-period = 20*6h = 5 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        pivot = pp_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: break above Donchian high + above pivot + volume confirmation
            if (price > donch_high[i] and 
                price > pivot and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: break below Donchian low + below pivot + volume confirmation
            elif (price < donch_low[i] and 
                  price < pivot and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low or below pivot
            if (price < donch_low[i] or
                price < pivot):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high or above pivot
            if (price > donch_high[i] or
                price > pivot):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Donchian_Pivot_Volume"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-13 22:28

# Strategy: 4h_1d_donchian_breakout_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.272 | +35.3% | -13.4% | 96 | PASS |
| ETHUSDT | -0.292 | -2.4% | -28.5% | 106 | FAIL |
| SOLUSDT | 0.442 | +66.8% | -29.8% | 103 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.015 | -5.1% | -8.1% | 40 | FAIL |
| SOLUSDT | 0.486 | +15.0% | -10.2% | 32 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_1d_donchian_breakout_v1
# Hypothesis: 4-hour Donchian channel (20-period) breakout with daily trend filter and volume confirmation.
# Long when price breaks above 20-bar high with price > daily EMA50 and volume > 1.5x average.
# Short when price breaks below 20-bar low with price < daily EMA50 and volume > 1.5x average.
# Works in bull markets (breakouts above resistance) and bear markets (breakdowns below support).
# Daily EMA50 filter ensures alignment with higher timeframe trend, reducing counter-trend trades.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema = close_1d[49]  # Initialize with first 50-period average
        multiplier = 2 / (50 + 1)
        ema_50_1d[49] = ema
        for i in range(50, len(close_1d)):
            ema = (close_1d[i] - ema) * multiplier + ema
            ema_50_1d[i] = ema
    
    # Align daily EMA50 to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 20-period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Calculate Donchian channels (20-period high/low)
        if i >= 20:
            donchian_high = np.max(high[i-20:i])
            donchian_low = np.min(low[i-20:i])
        else:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or below 20-period low
            if close[i] <= donchian_low:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above 20-period high
            if close[i] >= donchian_high:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high with trend and volume filters
            if (close[i] > donchian_high and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > vol_ma_20[i] * 1.5):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low with trend and volume filters
            elif (close[i] < donchian_low and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > vol_ma_20[i] * 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-09 08:58

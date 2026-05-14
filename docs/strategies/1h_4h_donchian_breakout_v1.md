# Strategy: 1h_4h_donchian_breakout_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.710 | -11.7% | -19.3% | 487 | FAIL |
| ETHUSDT | -0.753 | -20.8% | -32.5% | 496 | FAIL |
| SOLUSDT | 0.119 | +23.3% | -24.7% | 477 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.473 | +13.8% | -11.7% | 149 | PASS |

## Code
```python
#!/usr/bin/env python3
# 1h_4h_donchian_breakout_v1
# Hypothesis: 1-hour Donchian channel (20-period) breakout with 4-hour trend filter and volume confirmation.
# Long when price breaks above 20-bar high with price > 4h EMA50 and volume > 1.5x average.
# Short when price breaks below 20-bar low with price < 4h EMA50 and volume > 1.5x average.
# Uses 4h EMA50 for trend alignment to reduce counter-trend trades.
# Position size fixed at 0.20 to control risk and limit trade frequency.
# Target: 15-37 trades/year (60-150 total over 4 years) by using tight entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_donchian_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        ema = close_4h[49]  # Initialize with first 50-period average
        multiplier = 2 / (50 + 1)
        ema_50_4h[49] = ema
        for i in range(50, len(close_4h)):
            ema = (close_4h[i] - ema) * multiplier + ema
            ema_50_4h[i] = ema
    
    # Align 4h EMA50 to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
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
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20[i]):
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
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price returns to or above 20-period high
            if close[i] >= donchian_high:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter long: price breaks above Donchian high with trend and volume filters
            if (close[i] > donchian_high and 
                close[i] > ema_50_4h_aligned[i] and 
                volume[i] > vol_ma_20[i] * 1.5):
                position = 1
                signals[i] = 0.20
            # Enter short: price breaks below Donchian low with trend and volume filters
            elif (close[i] < donchian_low and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume[i] > vol_ma_20[i] * 1.5):
                position = -1
                signals[i] = -0.20
    
    return signals
```

## Last Updated
2026-04-09 08:59

# Strategy: 6h_12h_volume_breakout_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.255 | +4.7% | -21.3% | 61 | FAIL |
| ETHUSDT | 0.330 | +43.2% | -14.3% | 57 | PASS |
| SOLUSDT | 0.431 | +65.5% | -25.9% | 53 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.295 | +10.8% | -10.0% | 21 | PASS |
| SOLUSDT | -0.117 | +1.9% | -15.6% | 19 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 6h_12h_volume_breakout_v1
# Hypothesis: 6-hour price breakouts with volume confirmation and 12-hour trend filter.
# Long when price breaks above 24-period high with price > 12h EMA50 and volume > 2x average.
# Short when price breaks below 24-period low with price < 12h EMA50 and volume > 2x average.
# Uses 12h EMA50 for trend alignment to reduce counter-trend trades.
# Volume filter requires 2x average volume to ensure institutional participation.
# Position size fixed at 0.25 to balance risk and reward.
# Target: 50-150 total trades over 4 years (12-37/year) with tight entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_volume_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema = close_12h[49]  # Initialize with first 50-period average
        multiplier = 2 / (50 + 1)
        ema_50_12h[49] = ema
        for i in range(50, len(close_12h)):
            ema = (close_12h[i] - ema) * multiplier + ema
            ema_50_12h[i] = ema
    
    # Align 12h EMA50 to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: 24-period average
    vol_ma_24 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 24:
            vol_sum -= volume[i-24]
        if i >= 23:
            vol_ma_24[i] = vol_sum / 24
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_24[i]):
            signals[i] = 0.0
            continue
        
        # Calculate price channels (24-period high/low)
        if i >= 24:
            channel_high = np.max(high[i-24:i])
            channel_low = np.min(low[i-24:i])
        else:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or below 24-period low
            if close[i] <= channel_low:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above 24-period high
            if close[i] >= channel_high:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above channel high with trend and volume filters
            if (close[i] > channel_high and 
                close[i] > ema_50_12h_aligned[i] and 
                volume[i] > vol_ma_24[i] * 2.0):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below channel low with trend and volume filters
            elif (close[i] < channel_low and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume[i] > vol_ma_24[i] * 2.0):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-09 09:00

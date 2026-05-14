# Strategy: 6h_1d_volume_breakout_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.119 | +13.5% | -12.1% | 157 | FAIL |
| ETHUSDT | 0.432 | +48.2% | -17.2% | 130 | PASS |
| SOLUSDT | 0.931 | +147.0% | -20.0% | 116 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.928 | +22.7% | -7.8% | 43 | PASS |
| SOLUSDT | -0.646 | -6.3% | -21.0% | 40 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 6h_1d_volume_breakout_v2
# Hypothesis: 6-hour breakouts at daily high/low levels with volume confirmation (>2.0x 20-bar average volume).
# Daily high/low levels act as strong support/resistance; breaks signal momentum continuation.
# Designed for 6h timeframe to capture medium-term moves with controlled trade frequency (target: 20-35/year).
# Works in bull markets (upward breaks above daily high) and bear markets (downward breaks below daily low).
# Uses daily data for support/resistance levels, avoiding look-ahead bias via mtf_data helpers.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_volume_breakout_v2"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily high and low
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Align daily high/low to 6h timeframe
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, daily_low)
    
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
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(daily_high_aligned[i]) or np.isnan(daily_low_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or below daily low
            if close[i] <= daily_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above daily high
            if close[i] >= daily_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above daily high with volume confirmation
            if close[i] > daily_high_aligned[i] and volume[i] > vol_ma_20[i] * 2.0:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below daily low with volume confirmation
            elif close[i] < daily_low_aligned[i] and volume[i] > vol_ma_20[i] * 2.0:
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-09 08:57

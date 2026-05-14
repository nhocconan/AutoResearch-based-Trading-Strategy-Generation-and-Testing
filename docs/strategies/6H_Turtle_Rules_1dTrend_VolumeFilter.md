# Strategy: 6H_Turtle_Rules_1dTrend_VolumeFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.419 | +42.4% | -11.8% | 89 | PASS |
| ETHUSDT | 0.057 | +21.3% | -13.7% | 98 | PASS |
| SOLUSDT | 0.584 | +84.5% | -26.9% | 94 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.589 | -1.0% | -8.4% | 33 | FAIL |
| ETHUSDT | 0.277 | +10.0% | -8.5% | 32 | PASS |
| SOLUSDT | -0.552 | -5.1% | -15.8% | 34 | FAIL |

## Code
```python
#!/usr/bin/env python3
name = "6H_Turtle_Rules_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1-day EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1-day EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for 20-period high/low
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if EMA data not ready
        if np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate 20-period high and low for breakout levels
        period_high = np.max(high[i-20:i])
        period_low = np.min(low[i-20:i])
        
        # Determine trend
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average volume
        avg_volume = np.mean(volume[i-20:i])
        volume_confirm = volume[i] > avg_volume * 1.5
        
        if position == 0:
            # Enter long: price breaks above 20-period high + uptrend + volume confirmation
            if close[i] > period_high and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 20-period low + downtrend + volume confirmation
            elif close[i] < period_low and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 10-period low (Turtle exit rule)
            exit_low = np.min(low[i-10:i])
            if close[i] < exit_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above 10-period high
            exit_high = np.max(high[i-10:i])
            if close[i] > exit_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-09 22:09

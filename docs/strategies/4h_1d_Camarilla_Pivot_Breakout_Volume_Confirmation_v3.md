# Strategy: 4h_1d_Camarilla_Pivot_Breakout_Volume_Confirmation_v3

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 1.370 | +164.5% | -9.8% | 242 | PASS |
| ETHUSDT | 0.009 | +14.4% | -23.1% | 254 | PASS |
| SOLUSDT | 1.134 | +293.5% | -22.0% | 205 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.666 | -3.8% | -11.5% | 96 | FAIL |
| ETHUSDT | 0.049 | +5.3% | -17.2% | 86 | PASS |
| SOLUSDT | 0.862 | +28.3% | -17.8% | 88 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Breakout_Volume_Confirmation_v3
Hypothesis: Daily Camarilla pivot levels (S3/R3) provide strong support/resistance.
Breakouts above R3 or below S3 on 4h chart with volume expansion capture institutional moves.
Adds volume confirmation to reduce false breakouts. Works in both bull and bear markets
by trading breakouts regardless of direction. Target: 20-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each daily bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas
    close_prev = np.roll(close_1d, 1)
    close_prev[0] = close_1d[0]  # first bar uses its own close
    
    range_1d = high_1d - low_1d
    
    # Resistance levels (R3 used)
    R3 = close_prev + (range_1d * 1.2500 / 4)
    
    # Support levels (S3 used)
    S3 = close_prev - (range_1d * 1.2500 / 4)
    
    # Align levels to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(200, n):
        # Skip if any required data is not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long breakout: price breaks above R3 with volume expansion
        long_breakout = close[i] > R3_aligned[i] and volume_expansion[i]
        
        # Short breakdown: price breaks below S3 with volume expansion
        short_breakout = close[i] < S3_aligned[i] and volume_expansion[i]
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_1d_Camarilla_Pivot_Breakout_Volume_Confirmation_v3"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-13 19:17

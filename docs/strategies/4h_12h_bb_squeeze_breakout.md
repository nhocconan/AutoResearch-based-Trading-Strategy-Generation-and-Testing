# Strategy: 4h_12h_bb_squeeze_breakout

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.703 | -2.0% | -21.3% | 55 | FAIL |
| ETHUSDT | 0.366 | +40.9% | -16.2% | 56 | PASS |
| SOLUSDT | 0.260 | +37.4% | -22.1% | 47 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.320 | +9.4% | -7.8% | 16 | PASS |
| SOLUSDT | 0.603 | +14.6% | -6.7% | 18 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_12h_bb_squeeze_breakout
Uses Bollinger Bands squeeze on 12h to detect low volatility, then breaks out on 4h with volume confirmation.
Long when price breaks above upper BB after squeeze, short when breaks below lower BB.
Exit when price returns to middle band or volatility expands.
Designed for low trade frequency (target: 20-40 trades/year) to minimize fee drag.
Works in both trending and ranging markets by combining volatility contraction with breakout logic.
"""

name = "4h_12h_bb_squeeze_breakout"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 12h data for Bollinger Bands calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Bollinger Bands (20, 2) on 12h
    bb_length = 20
    bb_mult = 2.0
    
    # Middle band (SMA)
    basis = pd.Series(close_12h).rolling(window=bb_length, min_periods=bb_length).mean().values
    
    # Standard deviation
    dev = bb_mult * pd.Series(close_12h).rolling(window=bb_length, min_periods=bb_length).std().values
    
    # Upper and lower bands
    upper = basis + dev
    lower = basis - dev
    
    # Bandwidth for squeeze detection
    bandwidth = (upper - lower) / basis
    bandwidth_smoothed = pd.Series(bandwidth).rolling(window=5, min_periods=5).mean().values
    
    # Squeeze condition: bandwidth below 20-period lowest (low volatility)
    bandwidth_lowest = pd.Series(bandwidth_smoothed).rolling(window=20, min_periods=20).min().values
    squeeze = bandwidth_smoothed <= bandwidth_lowest
    
    # Align Bollinger Bands and squeeze to 4h
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower)
    basis_aligned = align_htf_to_ltf(prices, df_12h, basis)
    squeeze_aligned = align_htf_to_ltf(prices, df_12h, squeeze)
    
    # Volume confirmation on 4h: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(basis_aligned[i]) or np.isnan(squeeze_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above upper BB after squeeze, with volume
        if squeeze_aligned[i] and close[i] > upper_aligned[i] and vol_confirm[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short entry: price breaks below lower BB after squeeze, with volume
        elif squeeze_aligned[i] and close[i] < lower_aligned[i] and vol_confirm[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions
        elif position == 1 and close[i] <= basis_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= basis_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals
```

## Last Updated
2026-04-12 16:03

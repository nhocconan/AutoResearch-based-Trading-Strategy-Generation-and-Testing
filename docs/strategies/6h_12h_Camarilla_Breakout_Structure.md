# Strategy: 6h_12h_Camarilla_Breakout_Structure

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.305 | +39.9% | -12.6% | 45 | PASS |
| ETHUSDT | -0.014 | +12.7% | -23.9% | 44 | FAIL |
| SOLUSDT | 0.884 | +185.9% | -32.2% | 30 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.026 | +5.8% | -6.6% | 14 | PASS |
| SOLUSDT | -0.043 | +2.4% | -19.8% | 9 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h_12h_Camarilla_Breakout_Structure
Hypothesis: Camarilla pivot levels from 12h timeframe provide strong intraday support/resistance.
Breakouts above R3 or below S3 with volume confirmation indicate institutional participation,
while closes back inside the H-L range suggest fakeouts. The 1d EMA50 filter ensures trades
align with the medium-term trend, reducing whipsaws in ranging markets. This structure works
in both bull (breakouts continue) and bear (fades at resistance) markets by using price
action confirmation rather than pure breakout logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar: H-L range based
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    range_12h = high_12h - low_12h
    r4_12h = close_12h + 1.5 * range_12h
    r3_12h = close_12h + 1.1 * range_12h
    s3_12h = close_12h - 1.1 * range_12h
    s4_12h = close_12h - 1.5 * range_12h
    
    # Align 12h Camarilla levels to 6h
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or 
            np.isnan(r4_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Breakout above R3 with volume expansion
        # 2. OR re-entry after fakeout: price closes back above S3 after being below it
        # 3. Must be above 1d EMA50 for trend alignment
        breakout_long = (close[i] > r3_12h_aligned[i]) and volume_expansion[i]
        reentry_long = (close[i] > s3_12h_aligned[i]) and (low[i] <= s3_12h_aligned[i]) and volume_expansion[i]
        long_condition = (breakout_long or reentry_long) and (close[i] > ema_50_aligned[i])
        
        # Short conditions:
        # 1. Breakdown below S3 with volume expansion
        # 2. OR re-entry after fakeout: price closes back below R3 after being above it
        # 3. Must be below 1d EMA50 for trend alignment
        breakdown_short = (close[i] < s3_12h_aligned[i]) and volume_expansion[i]
        reentry_short = (close[i] < r3_12h_aligned[i]) and (high[i] >= r3_12h_aligned[i]) and volume_expansion[i]
        short_condition = (breakdown_short or reentry_short) and (close[i] < ema_50_aligned[i])
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "6h_12h_Camarilla_Breakout_Structure"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-13 18:26

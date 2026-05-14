# Strategy: 4H_Camarilla_R3_S3_1DTrend_With_Expiry

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.265 | +27.7% | -3.9% | 291 | PASS |
| ETHUSDT | 0.047 | +22.6% | -5.7% | 276 | PASS |
| SOLUSDT | -0.238 | +9.9% | -9.1% | 253 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.309 | -5.3% | -5.8% | 116 | FAIL |
| ETHUSDT | 0.372 | +9.1% | -3.9% | 99 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4H_Camarilla_R3_S3_1DTrend_With_Expiry
# Hypothesis: Uses Camarilla R3/S3 from daily timeframe with 1-day EMA34 trend filter and volume spike confirmation.
# Adds time-based exit (max 3 bars held) to prevent overtrading and improve performance in both bull and bear markets.
# Designed for low trade frequency (<40/year) with clear entry/exit rules.

name = "4H_Camarilla_R3_S3_1DTrend_With_Expiry"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily OHLC for Camarilla pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    rng = high_1d - low_1d
    camarilla_r3 = close_1d + rng * 1.1 / 4
    camarilla_s3 = close_1d - rng * 1.1 / 4
    
    # 1-day EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Camarilla levels and EMA to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 2.0x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_held = 0  # Track bars held in current position
    
    start_idx = 20  # Ensure we have volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_held = 0
            continue
        
        # Volume filter: spike confirmation
        volume_filter = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above R3 + Uptrend (price > EMA34) + volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema34_aligned[i] and
                volume_filter):
                signals[i] = 0.25
                position = 1
                bars_held = 1
            # Short: Price breaks below S3 + Downtrend (price < EMA34) + volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema34_aligned[i] and
                  volume_filter):
                signals[i] = -0.25
                position = -1
                bars_held = 1
        elif position != 0:
            # Increment bars held
            bars_held += 1
            
            # Exit conditions:
            # 1. Price returns inside pivot range (reversion to mean)
            # 2. Maximum hold time exceeded (3 bars)
            price_inside = (close[i] < r3_aligned[i] and close[i] > s3_aligned[i])
            time_exit = bars_held >= 3
            
            if price_inside or time_exit:
                signals[i] = 0.0
                position = 0
                bars_held = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals
```

## Last Updated
2026-05-07 03:22

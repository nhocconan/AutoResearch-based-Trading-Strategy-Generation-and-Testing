# Strategy: 4H_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.014 | +19.8% | -11.1% | 293 | FAIL |
| ETHUSDT | 0.466 | +45.4% | -8.4% | 266 | PASS |
| SOLUSDT | 0.862 | +104.2% | -15.7% | 220 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.734 | +16.4% | -6.0% | 96 | PASS |
| SOLUSDT | 0.782 | +17.3% | -8.9% | 81 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4H_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike_v1
# Hypothesis: Combines Camarilla pivot levels (R3/S3) breakout with 12h EMA trend filter and volume spike confirmation. Designed for 4h timeframe to capture medium-term breakouts with low trade frequency (~25-40 trades/year). The Camarilla levels provide statistically significant support/resistance, EMA filter ensures trend alignment, and volume spike confirms institutional participation. Works in both bull and bear markets by following the trend.

name = "4H_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike_v1"
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
    
    # Get 12h data for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 12h bar
    # Camarilla: R4 = C + ((H-L) * 1.5000), R3 = C + ((H-L) * 1.2500), etc.
    # We use R3 and S3 as key levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate R3 and S3 for each 12h bar
    # R3 = close + (high-low)*1.1/4
    # S3 = close - (high-low)*1.1/4
    r3_12h = close_12h + (high_12h - low_12h) * 1.1 / 4
    s3_12h = close_12h - (high_12h - low_12h) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    
    # Get 12h EMA for trend filter
    ema12_12h = pd.Series(close_12h).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema12_12h_aligned = align_htf_to_ltf(prices, df_12h, ema12_12h)
    
    # Volume spike: current volume > 2x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure we have volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or 
            np.isnan(ema12_12h_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation
        volume_filter = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R3 + uptrend + volume spike
            if (close[i] > r3_12h_aligned[i] and 
                close[i] > ema12_12h_aligned[i] and
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + downtrend + volume spike
            elif (close[i] < s3_12h_aligned[i] and 
                  close[i] < ema12_12h_aligned[i] and
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns below EMA or to S3 level (mean reversion)
            if close[i] < ema12_12h_aligned[i] or close[i] < s3_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns above EMA or to R3 level (mean reversion)
            if close[i] > ema12_12h_aligned[i] or close[i] > r3_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-07 03:09

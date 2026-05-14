# Strategy: 4h_Camarilla_R3S3_12hEMA50_VolumeSpike_TrendFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.283 | +32.8% | -5.9% | 167 | PASS |
| ETHUSDT | 0.242 | +32.8% | -11.1% | 160 | PASS |
| SOLUSDT | 0.233 | +34.6% | -18.3% | 129 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.976 | -3.2% | -7.4% | 58 | FAIL |
| ETHUSDT | 0.695 | +16.6% | -7.1% | 56 | PASS |
| SOLUSDT | 0.229 | +8.9% | -10.7% | 41 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation
# Uses Camarilla pivot levels from 4h chart for structure, 12h EMA50 for trend filter, and volume spike for confirmation.
# Designed for 19-50 trades/year to minimize fee drag. Works in bull markets via breakout continuations and in bear markets via breakdown continuations.
# The 12h EMA50 provides a smooth trend filter that adapts to changing regimes while avoiding whipsaw.
# BTC and ETH are primary targets; SOL is secondary.

name = "4h_Camarilla_R3S3_12hEMA50_VolumeSpike_TrendFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter and Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h EMA50 trend filter from prior completed 12h bar
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_shifted = np.roll(ema50_12h, 1)
    ema50_12h_shifted[0] = np.nan
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h_shifted)
    
    # Calculate Camarilla levels (R3, S3) from prior completed 12h bar
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    camarilla_range = high_12h - low_12h
    r3 = close_12h + 1.1 * camarilla_range / 2
    s3 = close_12h - 1.1 * camarilla_range / 2
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND above 12h EMA50 AND volume spike
            if close[i] > r3_aligned[i] and close[i] > ema50_12h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 AND below 12h EMA50 AND volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema50_12h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below S3 OR below 12h EMA50
            if close[i] < s3_aligned[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above R3 OR above 12h EMA50
            if close[i] > r3_aligned[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-04 14:01

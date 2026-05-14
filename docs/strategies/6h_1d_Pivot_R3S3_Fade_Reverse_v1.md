# Strategy: 6h_1d_Pivot_R3S3_Fade_Reverse_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.010 | +20.4% | -10.4% | 189 | PASS |
| ETHUSDT | 0.436 | +45.1% | -10.7% | 173 | PASS |
| SOLUSDT | 0.392 | +50.8% | -18.5% | 139 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.990 | -11.4% | -14.7% | 77 | FAIL |
| ETHUSDT | 1.310 | +28.0% | -7.1% | 61 | PASS |
| SOLUSDT | -0.334 | +0.2% | -11.7% | 53 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 6h_1d_Pivot_R3S3_Fade_Reverse_v1
# Hypothesis: On 6h timeframe, trade reversals at 1d Camarilla R3/S3 levels with volume confirmation.
# In ranging markets, price tends to reverse at R3/S3; in trending markets, breaks through R4/S4.
# Uses 1d ADX to filter ranging (ADX < 25) for reversals and trending (ADX > 25) for breakouts.
# Targets 15-35 trades/year by requiring confluence of level, volume, and regime filter.

name = "6h_1d_Pivot_R3S3_Fade_Reverse_v1"
timeframe = "6h"
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    
    # Pivot point and ranges
    pivot_1d = typical_price_1d
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R3, S3, R4, S4
    r3_1d = close_1d + (range_1d * 1.1 / 6)
    s3_1d = close_1d - (range_1d * 1.1 / 6)
    r4_1d = close_1d + (range_1d * 1.1 / 4)
    s4_1d = close_1d - (range_1d * 1.1 / 4)
    
    # Align 1d levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate 1d ADX for trend/ranging filter (14-period)
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR and DM
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[1:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr = smooth_wilder(tr, 14)
    plus_di = 100 * smooth_wilder(plus_dm, 14) / atr
    minus_di = 100 * smooth_wilder(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smooth_wilder(dx, 14)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Ranging market (ADX < 25): fade at R3/S3
            if adx_aligned[i] < 25:
                # Long near S3 with volume confirmation
                if (close[i] <= s3_aligned[i] * 1.005 and 
                    close[i] >= s3_aligned[i] * 0.995 and
                    volume[i] > 1.5 * volume_ma[i]):
                    signals[i] = 0.25
                    position = 1
                # Short near R3 with volume confirmation
                elif (close[i] >= r3_aligned[i] * 0.995 and 
                      close[i] <= r3_aligned[i] * 1.005 and
                      volume[i] > 1.5 * volume_ma[i]):
                    signals[i] = -0.25
                    position = -1
            # Trending market (ADX > 25): breakout at R4/S4
            elif adx_aligned[i] > 25:
                # Long breakout above R4 with volume
                if (close[i] > r4_aligned[i] * 1.005 and 
                    volume[i] > 2.0 * volume_ma[i]):
                    signals[i] = 0.25
                    position = 1
                # Short breakdown below S4 with volume
                elif (close[i] < s4_aligned[i] * 0.995 and 
                      volume[i] > 2.0 * volume_ma[i]):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: reverse at opposite level or ADX shifts to ranging
            if (adx_aligned[i] < 25 and close[i] >= r3_aligned[i] * 0.995) or \
               (adx_aligned[i] > 25 and close[i] < s4_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: reverse at opposite level or ADX shifts to ranging
            if (adx_aligned[i] < 25 and close[i] <= s3_aligned[i] * 1.005) or \
               (adx_aligned[i] > 25 and close[i] > r4_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-20 02:15

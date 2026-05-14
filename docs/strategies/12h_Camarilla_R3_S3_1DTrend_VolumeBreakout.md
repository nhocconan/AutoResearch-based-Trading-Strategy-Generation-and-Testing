# Strategy: 12h_Camarilla_R3_S3_1DTrend_VolumeBreakout

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.200 | +28.2% | -6.6% | 70 | PASS |
| ETHUSDT | 0.074 | +23.4% | -7.2% | 64 | PASS |
| SOLUSDT | 0.094 | +23.1% | -21.7% | 61 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.602 | +1.1% | -5.8% | 25 | FAIL |
| ETHUSDT | 0.051 | +6.3% | -7.6% | 26 | PASS |
| SOLUSDT | -0.006 | +5.5% | -7.7% | 22 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_1DTrend_VolumeBreakout
# Hypothesis: 12-hour timeframe strategy using daily Camarilla R3/S3 breakouts with 1-day EMA34 trend filter and volume spike confirmation. 
# Targets fewer trades (12-37/year) to reduce fee drag while maintaining edge in bull/bear markets via trend alignment and volume confirmation.
# Uses proper 12h/1d multi-timeframe alignment to avoid look-ahead.

name = "12h_Camarilla_R3_S3_1DTrend_VolumeBreakout"
timeframe = "12h"
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
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough data for EMA34
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels for each day
    range_1d = prev_high - prev_low
    r3 = prev_close + range_1d * 1.1 / 4
    s3 = prev_close - range_1d * 1.1 / 4
    pp = (prev_high + prev_low + prev_close) / 3  # Pivot point
    
    # Calculate 1-day EMA34 for trend filter
    ema_34 = pd.Series(prev_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Camarilla levels, EMA, and pivot to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume filter: current volume > 2.0x average volume (24-period) - balanced for 12h
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 24)  # Ensure we have EMA34 and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(ema_34_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation (2.0x average volume)
        volume_filter = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above R3 + uptrend (price > EMA34) + volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_aligned[i] and   # Uptrend filter
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 + downtrend (price < EMA34) + volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_aligned[i] and   # Downtrend filter
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit conditions:
            # 1. Price returns to pivot point (mean reversion)
            # 2. Opposite Camarilla level break (trend exhaustion)
            at_pivot = abs(close[i] - pp_aligned[i]) < (r3_aligned[i] - pp_aligned[i]) * 0.1  # Within 10% of PP
            opposite_break = (position == 1 and close[i] < s3_aligned[i]) or \
                           (position == -1 and close[i] > r3_aligned[i])
            
            if at_pivot or opposite_break:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals
```

## Last Updated
2026-05-07 03:30

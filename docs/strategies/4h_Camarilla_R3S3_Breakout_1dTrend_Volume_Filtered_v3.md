# Strategy: 4h_Camarilla_R3S3_Breakout_1dTrend_Volume_Filtered_v3

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.091 | +24.2% | -7.6% | 58 | PASS |
| ETHUSDT | 0.033 | +21.2% | -13.0% | 57 | PASS |
| SOLUSDT | 0.440 | +52.0% | -19.7% | 38 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.311 | +3.4% | -7.7% | 18 | FAIL |
| ETHUSDT | 0.706 | +15.1% | -6.8% | 19 | PASS |
| SOLUSDT | -0.335 | +1.8% | -9.7% | 19 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dTrend_Volume_Filtered_v3
Hypothesis: Reduce trade frequency by requiring volume > 2.5x average (tighter than v2) and adding a 1-day ADX > 25 trend filter to avoid chop. Uses 1d Camarilla R3/S3 for breakout levels, 1d ADX for trend strength, and volume spike for confirmation. Designed for 15-25 trades/year to avoid fee drag while maintaining edge in both bull and bear markets.
"""
name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume_Filtered_v3"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and Camarilla pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day OHLC for Camarilla pivot
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot levels calculation
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r3_1d = close_1d + (range_1d * 1.1 / 2)
    s3_1d = close_1d - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # 1-day ADX for trend filter (requires trend strength)
    # Calculate True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_period = 14
    atr = np.full(len(tr), np.nan)
    dm_plus_smooth = np.full(len(dm_plus), np.nan)
    dm_minus_smooth = np.full(len(dm_minus), np.nan)
    
    # Wilder's smoothing
    for i in range(len(tr)):
        if i < tr_period:
            continue
        if i == tr_period:
            atr[i] = np.nansum(tr[i-tr_period+1:i+1])
            dm_plus_smooth[i] = np.nansum(dm_plus[i-tr_period+1:i+1])
            dm_minus_smooth[i] = np.nansum(dm_minus[i-tr_period+1:i+1])
        else:
            atr[i] = atr[i-1] - (atr[i-1] / tr_period) + tr[i]
            dm_plus_smooth[i] = dm_plus_smooth[i-1] - (dm_plus_smooth[i-1] / tr_period) + dm_plus[i]
            dm_minus_smooth[i] = dm_minus_smooth[i-1] - (dm_minus_smooth[i-1] / tr_period) + dm_minus[i]
    
    # Calculate DI and DX
    di_plus = np.full(len(tr), np.nan)
    di_minus = np.full(len(tr), np.nan)
    dx = np.full(len(tr), np.nan)
    
    for i in range(tr_period, len(tr)):
        if atr[i] > 0:
            di_plus[i] = 100 * (dm_plus_smooth[i] / atr[i])
            di_minus[i] = 100 * (dm_minus_smooth[i] / atr[i])
            dx[i] = 100 * (np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i]))
    
    # Calculate ADX (smoothed DX)
    adx = np.full(len(tr), np.nan)
    adx_period = 14
    for i in range(tr_period + adx_period - 1, len(tr)):
        if i == tr_period + adx_period - 1:
            adx[i] = np.nanmean(dx[tr_period:i+1])
        else:
            adx[i] = (adx[i-1] * (adx_period - 1) + dx[i]) / adx_period
    
    adx_1d = adx
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume filter: current volume > 2.5 * 100-period average (tighter than v2)
    vol_avg = pd.Series(volume).rolling(window=100, min_periods=100).mean().values
    volume_filter = volume > (vol_avg * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup for ADX and volume average
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_avg[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 + 1d strong uptrend (ADX > 25) + volume filter
            if (close[i] > r3_1d_aligned[i] and 
                adx_1d_aligned[i] > 25 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + 1d strong downtrend (ADX > 25) + volume filter
            elif (close[i] < s3_1d_aligned[i] and 
                  adx_1d_aligned[i] > 25 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to opposite Camarilla level (S3 for long, R3 for short)
            if position == 1:
                if close[i] <= s3_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] >= r3_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-07 07:15

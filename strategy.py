#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Volume_Trend_v1
Hypothesis: On 4h timeframe, use 1d Camarilla pivot levels for mean reversion with volume confirmation and ADX trend filter.
Buy when price touches S3 support with volume spike in a low-volatility regime (ADX < 25).
Sell when price touches R3 resistance with volume spike in a low-volatility regime (ADX < 25).
Exit when price reverts to the previous day's close (pivot point) or ADX rises above 25 indicating trend.
Designed for 4h to capture reversals in ranging markets while avoiding trending conditions that cause false reversals.
Works in both bull and bear markets by adapting to volatility regime.
"""

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
    
    # Load 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_s3 = np.full_like(close_1d, np.nan)
    camarilla_r3 = np.full_like(close_1d, np.nan)
    camarilla_pivot = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        if i == 0 or np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i]):
            continue
        # Use previous day's data for today's levels
        if i > 0:
            phigh = high_1d[i-1]
            plow = low_1d[i-1]
            pclose = close_1d[i-1]
            if not (np.isnan(phigh) or np.isnan(plow) or np.isnan(pclose)):
                range_val = phigh - plow
                camarilla_pivot[i] = (phigh + plow + pclose) / 3
                camarilla_s3[i] = pclose - range_val * 1.1 / 2
                camarilla_r3[i] = pclose + range_val * 1.1 / 2
    
    # Load 1d data for ADX calculation
    if len(high_1d) < 14:
        return np.zeros(n)
    
    # Calculate ADX on 1d data
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(high_1d)
    tr = np.zeros_like(high_1d)
    
    for i in range(1, len(high_1d)):
        if np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(high_1d[i-1]) or np.isnan(low_1d[i-1]):
            continue
        high_diff = high_1d[i] - high_1d[i-1]
        low_diff = low_1d[i-1] - low_1d[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high_1d[i] - low_1d[i], 
                   abs(high_1d[i] - high_1d[i-1]), 
                   abs(low_1d[i] - low_1d[i-1]))
    
    # Smooth TR, +DM, -DM with Wilder's smoothing (alpha = 1/14)
    atr = np.zeros_like(high_1d)
    plus_di = np.zeros_like(high_1d)
    minus_di = np.zeros_like(high_1d)
    dx = np.zeros_like(high_1d)
    adx = np.full_like(high_1d, np.nan)
    
    if len(high_1d) >= 14:
        # Initial values
        atr[13] = np.nansum(tr[1:14])
        plus_dm_sum = np.nansum(plus_dm[1:14])
        minus_dm_sum = np.nansum(minus_dm[1:14])
        
        for i in range(14, len(high_1d)):
            if np.isnan(tr[i]) or np.isnan(plus_dm[i]) or np.isnan(minus_dm[i]):
                atr[i] = atr[i-1]
                plus_dm_sum = plus_dm_sum
                minus_dm_sum = minus_dm_sum
            else:
                atr[i] = (atr[i-1] * 13 + tr[i]) / 14
                plus_dm_sum = (plus_dm_sum * 13 + plus_dm[i]) / 14
                minus_dm_sum = (minus_dm_sum * 13 + minus_dm[i]) / 14
            
            if atr[i] > 0:
                plus_di[i] = 100 * plus_dm_sum / atr[i]
                minus_di[i] = 100 * minus_dm_sum / atr[i]
                if plus_di[i] + minus_di[i] > 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        # Calculate ADX as smoothed DX
        if len(high_1d) >= 27:
            adx[26] = np.nanmean(dx[14:27])
            for i in range(27, len(high_1d)):
                if np.isnan(dx[i]):
                    adx[i] = adx[i-1]
                else:
                    adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align Camarilla levels and ADX to 4h timeframe
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # Start after enough data for alignment
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 4h volume vs 20-period average
        vol_ma_20 = np.full_like(volume, np.nan)
        for j in range(19, len(volume)):
            vol_ma_20[j] = np.mean(volume[j-19:j+1])
        vol_ma_20_aligned = vol_ma_20  # Already on 4h timeframe
        
        if np.isnan(vol_ma_20_aligned[i]) or vol_ma_20_aligned[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for long entries: price touches S3 with volume spike in low volatility regime
            if (close[i] <= camarilla_s3_aligned[i] * 1.001 and  # Allow small tolerance
                volume_ratio > 1.8 and
                adx_aligned[i] < 25):
                position = 1
                signals[i] = position_size
            # Look for short entries: price touches R3 with volume spike in low volatility regime
            elif (close[i] >= camarilla_r3_aligned[i] * 0.999 and  # Allow small tolerance
                  volume_ratio > 1.8 and
                  adx_aligned[i] < 25):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches pivot point or ADX rises indicating trend
            if (close[i] >= camarilla_pivot_aligned[i] * 0.999 or
                adx_aligned[i] >= 25):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches pivot point or ADX rises indicating trend
            if (close[i] <= camarilla_pivot_aligned[i] * 1.001 or
                adx_aligned[i] >= 25):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Camarilla_Pivot_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0
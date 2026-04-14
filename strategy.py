#!/usr/bin/env python3
"""
6h_1d_ADX_RangeBreakout_v1
Hypothesis: On 6h timeframe, breakout from 1d range (high-low) with ADX filter to avoid false breakouts in chop.
Long when price breaks above 1d high with volume spike and ADX > 25 (trending).
Short when price breaks below 1d low with volume spike and ADX > 25.
Exit when price returns to 1d midpoint or ADX drops below 20 (range resumption).
Works in bull/bear by capturing breakouts in trending regimes while avoiding whipsaws in ranges.
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
    
    # Load 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d range and midpoint
    range_1d = high_1d - low_1d
    midpoint_1d = (high_1d + low_1d) / 2
    
    # Calculate ADX on 1d data (14 period)
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
    
    # Wilder's smoothing
    atr = np.zeros_like(high_1d)
    plus_di = np.zeros_like(high_1d)
    minus_di = np.zeros_like(high_1d)
    dx = np.zeros_like(high_1d)
    adx = np.full_like(high_1d, np.nan)
    
    if len(high_1d) >= 14:
        # Initial smoothed values
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
    
    # Align 1d data to 6h
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    midpoint_1d_aligned = align_htf_to_ltf(prices, df_1d, midpoint_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume ratio (6h volume vs 20-period average)
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position
    
    for i in range(20, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_1d_aligned[i]) or np.isnan(low_1d_aligned[i]) or
            np.isnan(midpoint_1d_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        if position == 0:
            # Long: break above 1d high with volume and trend
            if (close[i] > high_1d_aligned[i] and
                vol_ratio > 2.0 and
                adx_aligned[i] > 25):
                position = 1
                signals[i] = position_size
            # Short: break below 1d low with volume and trend
            elif (close[i] < low_1d_aligned[i] and
                  vol_ratio > 2.0 and
                  adx_aligned[i] > 25):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: return to midpoint or trend weakens
            if (close[i] < midpoint_1d_aligned[i] or
                adx_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: return to midpoint or trend weakens
            if (close[i] > midpoint_1d_aligned[i] or
                adx_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_ADX_RangeBreakout_v1"
timeframe = "6h"
leverage = 1.0
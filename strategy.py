#!/usr/bin/env python3
"""
4h_1d_Range_Breakout_With_Volume_Confirmation_v1
Hypothesis: On 4h timeframe, buy when price breaks above the previous day's high with volume confirmation in a low-volatility regime (ADX < 25), and sell when price breaks below the previous day's low with volume confirmation in a low-volatility regime. Exit when price returns to the previous day's close or ADX rises above 25 indicating a new trend.
This strategy captures breakouts from the previous day's range, which often occur in both bull and bear markets as price seeks liquidity. Volume confirmation ensures genuine breakouts, while the ADX filter avoids false signals in strong trends where breakouts may fail.
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
    
    # Load 1d data for previous day's high, low, and close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's high, low, and close for each day
    prev_high = np.full_like(high_1d, np.nan)
    prev_low = np.full_like(low_1d, np.nan)
    prev_close = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(high_1d)):
        prev_high[i] = high_1d[i-1]
        prev_low[i] = low_1d[i-1]
        prev_close[i] = close_1d[i-1]
    
    # Calculate ADX on 1d data
    if len(high_1d) < 14:
        return np.zeros(n)
    
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
    
    # Align 1d data to 4h timeframe
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Precompute 20-period volume moving average for efficiency
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    for i in range(20, n):  # Start after enough data for alignment
        # Skip if any critical data is NaN
        if (np.isnan(prev_high_aligned[i]) or np.isnan(prev_low_aligned[i]) or
            np.isnan(prev_close_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 4h volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        if position == 0:
            # Look for long entries: price breaks above previous day's high with volume confirmation in low volatility regime
            if (close[i] > prev_high_aligned[i] and
                volume_ratio > 1.8 and
                adx_aligned[i] < 25):
                position = 1
                signals[i] = position_size
            # Look for short entries: price breaks below previous day's low with volume confirmation in low volatility regime
            elif (close[i] < prev_low_aligned[i] and
                  volume_ratio > 1.8 and
                  adx_aligned[i] < 25):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to previous day's close or ADX rises indicating trend
            if (close[i] <= prev_close_aligned[i] or
                adx_aligned[i] >= 25):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to previous day's close or ADX rises indicating trend
            if (close[i] >= prev_close_aligned[i] or
                adx_aligned[i] >= 25):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Range_Breakout_With_Volume_Confirmation_v1"
timeframe = "4h"
leverage = 1.0
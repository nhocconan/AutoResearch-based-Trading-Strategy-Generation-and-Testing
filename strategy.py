#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for regime filter (ADX)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX on 12h data (Wilder smoothing)
    if len(high_12h) < 14:
        return np.zeros(n)
    
    plus_dm = np.zeros_like(high_12h)
    minus_dm = np.zeros_like(high_12h)
    tr = np.zeros_like(high_12h)
    
    for i in range(1, len(high_12h)):
        if np.isnan(high_12h[i]) or np.isnan(low_12h[i]) or np.isnan(high_12h[i-1]) or np.isnan(low_12h[i-1]):
            continue
        high_diff = high_12h[i] - high_12h[i-1]
        low_diff = low_12h[i-1] - low_12h[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high_12h[i] - low_12h[i], 
                   abs(high_12h[i] - high_12h[i-1]), 
                   abs(low_12h[i] - low_12h[i-1]))
    
    # Wilder smoothing (alpha = 1/14)
    atr = np.zeros_like(high_12h)
    plus_di = np.zeros_like(high_12h)
    minus_di = np.zeros_like(high_12h)
    dx = np.zeros_like(high_12h)
    adx = np.full_like(high_12h, np.nan)
    
    if len(high_12h) >= 14:
        # Initial values (first 14 periods)
        atr[13] = np.nansum(tr[1:14])
        plus_dm_sum = np.nansum(plus_dm[1:14])
        minus_dm_sum = np.nansum(minus_dm[1:14])
        
        for i in range(14, len(high_12h)):
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
        
        # Calculate ADX as smoothed DX (14-period)
        if len(high_12h) >= 27:
            adx[26] = np.nanmean(dx[14:27])
            for i in range(27, len(high_12h)):
                if np.isnan(dx[i]):
                    adx[i] = adx[i-1]
                else:
                    adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align 12h ADX to 4h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Load 1d data for entry levels (high/low)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Align 1d levels to 4h timeframe
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    
    # Calculate 4-period volume MA for volume confirmation
    vol_ma_4 = np.full_like(volume, np.nan)
    for i in range(3, len(volume)):
        vol_ma_4[i] = np.mean(volume[i-3:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # Conservative size to limit trades
    
    for i in range(20, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_1d_aligned[i]) or np.isnan(low_1d_aligned[i]) or
            np.isnan(adx_12h_aligned[i]) or np.isnan(vol_ma_4[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 4h volume vs 4-period average
        if vol_ma_4[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_4[i]
        
        if position == 0:
            # Look for long entries: breakout above 1d high with volume surge in strong trend
            if (close[i] > high_1d_aligned[i] and 
                volume_ratio > 2.0 and  # Volume surge
                adx_12h_aligned[i] > 25):   # Trending regime
                position = 1
                signals[i] = position_size
            # Look for short entries: breakdown below 1d low with volume surge in strong trend
            elif (close[i] < low_1d_aligned[i] and 
                  volume_ratio > 2.0 and
                  adx_12h_aligned[i] > 25):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 1d low or trend weakens
            if (close[i] < low_1d_aligned[i] or
                adx_12h_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above 1d high or trend weakens
            if (close[i] > high_1d_aligned[i] or
                adx_12h_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_ADX_Trend_1d_Breakout_v1"
timeframe = "4h"
leverage = 1.0
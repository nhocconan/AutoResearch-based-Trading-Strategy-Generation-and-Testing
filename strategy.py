#!/usr/bin/env python3
# 1d_weekly_donchian_volume_chop_v1
# Hypothesis: Daily Donchian(20) breakout with volume confirmation and weekly chop regime filter.
# Long: price breaks above 20-period daily high with volume > 1.8x average AND weekly chop < 61.8 (trending)
# Short: price breaks below 20-period daily low with volume > 1.8x average AND weekly chop < 61.8 (trending)
# Exit: price reverses to 10-period opposite Donchian level or chop > 61.8 (ranging)
# Uses 1d primary timeframe with 1w HTF for regime filter to reduce overtrading.
# Target: 30-100 trades over 4 years (7-25/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_donchian_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily Donchian channels (20-period)
    highest_20 = np.full(n, np.nan)
    lowest_20 = np.full(n, np.nan)
    for i in range(20, n):
        highest_20[i] = np.max(high[i-20:i])
        lowest_20[i] = np.min(low[i-20:i])
    
    # Calculate daily Donchian exit channels (10-period)
    highest_10 = np.full(n, np.nan)
    lowest_10 = np.full(n, np.nan)
    for i in range(10, n):
        highest_10[i] = np.max(high[i-10:i])
        lowest_10[i] = np.min(low[i-10:i])
    
    # Calculate volume ratio (current vs 20-period average)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    vol_ratio = np.where(vol_sma > 0, volume / vol_sma, 0)
    
    # Get weekly data for chop regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Chopiness Index on weekly data (14-period)
    chop_1w = np.full(len(df_1w), np.nan)
    for i in range(14, len(df_1w)):
        atr_sum = 0
        for j in range(i-13, i+1):
            tr = max(df_1w['high'].iloc[j] - df_1w['low'].iloc[j],
                     abs(df_1w['high'].iloc[j] - df_1w['close'].iloc[j-1]),
                     abs(df_1w['low'].iloc[j] - df_1w['close'].iloc[j-1]))
            atr_sum += tr
        atr = atr_sum / 14
        max_high = np.max(df_1w['high'].iloc[i-13:i+1].values)
        min_low = np.min(df_1w['low'].iloc[i-13:i+1].values)
        if max_high != min_low:
            chop_1w[i] = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
        else:
            chop_1w[i] = 50  # neutral when no range
    
    # Align weekly chop to daily timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        vol_r = vol_ratio[i]
        ch = chop_aligned[i]
        price = close[i]
        
        if np.isnan(vol_r) or np.isnan(ch):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        highest = highest_20[i]
        lowest = lowest_20[i]
        exit_high = highest_10[i]
        exit_low = lowest_10[i]
        
        if np.isnan(highest) or np.isnan(lowest):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            if price < exit_low or ch > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            if price > exit_high or ch > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if price > highest and vol_r > 1.8 and ch < 61.8:
                position = 1
                signals[i] = 0.25
            elif price < lowest and vol_r > 1.8 and ch < 61.8:
                position = -1
                signals[i] = -0.25
    
    return signals
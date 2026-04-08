#!/usr/bin/env python3
# 12h_daily_donchian_volume_chop_v1
# Hypothesis: 12h Donchian(20) breakout with volume confirmation and 1d chop regime filter.
# Long: price breaks above 20-period high with volume > 2.0x average AND 1d chop < 61.8 (trending)
# Short: price breaks below 20-period low with volume > 2.0x average AND 1d chop < 61.8 (trending)
# Exit: price reverses to 10-period opposite Donchian level or chop > 61.8 (ranging)
# Uses 12h primary timeframe with 1d HTF for regime filter to reduce overtrading.
# Target: 75-150 trades over 4 years (19-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_daily_donchian_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    highest_20 = np.full(n, np.nan)
    lowest_20 = np.full(n, np.nan)
    for i in range(20, n):
        highest_20[i] = np.max(high[i-20:i])
        lowest_20[i] = np.min(low[i-20:i])
    
    # Calculate 12h Donchian exit channels (10-period)
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
    
    # Get 1d data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Chopiness Index on 1d data (14-period)
    chop_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_sum = 0
        for j in range(i-13, i+1):
            tr = max(df_1d['high'].iloc[j] - df_1d['low'].iloc[j],
                     abs(df_1d['high'].iloc[j] - df_1d['close'].iloc[j-1]),
                     abs(df_1d['low'].iloc[j] - df_1d['close'].iloc[j-1]))
            atr_sum += tr
        atr = atr_sum / 14
        max_high = np.max(df_1d['high'].iloc[i-13:i+1].values)
        min_low = np.min(df_1d['low'].iloc[i-13:i+1].values)
        if max_high != min_low:
            chop_1d[i] = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
        else:
            chop_1d[i] = 50  # neutral when no range
    
    # Align 1d chop to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
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
            if price > highest and vol_r > 2.0 and ch < 61.8:
                position = 1
                signals[i] = 0.25
            elif price < lowest and vol_r > 2.0 and ch < 61.8:
                position = -1
                signals[i] = -0.25
    
    return signals
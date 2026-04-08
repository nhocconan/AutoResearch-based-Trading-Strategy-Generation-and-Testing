#!/usr/bin/env python3
# 4h_12h_donchian_volume_chop_v2
# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h chop regime filter.
# Long: price breaks above 20-period high with volume > 1.8x average AND 12h chop < 61.8 (trending)
# Short: price breaks below 20-period low with volume > 1.8x average AND 12h chop < 61.8 (trending)
# Exit: price reverses to 10-period opposite Donchian level or chop > 61.8 (ranging)
# Uses 4h primary timeframe with 12h HTF for regime filter to reduce overtrading.
# Target: 75-200 trades over 4 years (19-50/year) to minimize fee drag.
# Improved: Added proper min_periods and vectorized calculations for performance.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_donchian_volume_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian channels (20-period) using pandas rolling for efficiency
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    highest_20 = high_series.rolling(window=20, min_periods=20).max().values
    lowest_20 = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h Donchian exit channels (10-period)
    highest_10 = high_series.rolling(window=10, min_periods=10).max().values
    lowest_10 = low_series.rolling(window=10, min_periods=10).min().values
    
    # Calculate volume ratio (current vs 20-period average)
    vol_series = pd.Series(volume)
    vol_sma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_sma > 0, volume / vol_sma, 0)
    
    # Get 12h data for chop regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Chopiness Index on 12h data (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range calculation
    tr = np.zeros(len(df_12h))
    tr[0] = high_12h[0] - low_12h[0]
    for i in range(1, len(df_12h)):
        hl = high_12h[i] - low_12h[i]
        hc = abs(high_12h[i] - close_12h[i-1])
        lc = abs(low_12h[i] - close_12h[i-1])
        tr[i] = max(hl, hc, lc)
    
    # ATR calculation
    atr_12h = np.full(len(df_12h), np.nan)
    for i in range(13, len(df_12h)):  # min_periods=14
        atr_12h[i] = np.mean(tr[i-13:i+1])
    
    # Chop calculation
    chop_12h = np.full(len(df_12h), np.nan)
    for i in range(13, len(df_12h)):
        atr_sum = np.sum(tr[i-13:i+1])
        max_high = np.max(high_12h[i-13:i+1])
        min_low = np.min(low_12h[i-13:i+1])
        if max_high != min_low:
            chop_12h[i] = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
        else:
            chop_12h[i] = 50  # neutral when no range
    
    # Align 12h chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
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
        
        if np.isnan(highest) or np.isnan(lowest) or np.isnan(exit_high) or np.isnan(exit_low):
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
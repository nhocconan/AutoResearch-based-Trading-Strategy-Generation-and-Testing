#!/usr/bin/env python3
# 4h_donchian_hma_volume_chop_v1
# Hypothesis: 4h Donchian(20) breakout with volume confirmation and HMA(21) trend filter on 1d timeframe.
# Long: price breaks above 20-period high with volume > 1.8x average AND 1d HMA(21) > previous HMA(21) (uptrend)
# Short: price breaks below 20-period low with volume > 1.8x average AND 1d HMA(21) < previous HMA(21) (downtrend)
# Exit: price reverses to 10-period opposite Donchian level
# Uses 4h primary timeframe with 1d HTF for trend filter to reduce overtrading.
# Target: 75-200 trades over 4 years (19-50/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_hma_volume_chop_v1"
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
    
    # Calculate 4h Donchian channels (20-period)
    highest_20 = np.full(n, np.nan)
    lowest_20 = np.full(n, np.nan)
    for i in range(20, n):
        highest_20[i] = np.max(high[i-20:i])
        lowest_20[i] = np.min(low[i-20:i])
    
    # Calculate 4h Donchian exit channels (10-period)
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
    
    # Get 1d data for HMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate HMA(21) on 1d close
    close_1d = df_1d['close'].values
    half_length = 21 // 2
    sqrt_length = int(np.sqrt(21))
    
    # WMA function
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        wma_vals = np.full(len(values), np.nan)
        for i in range(window - 1, len(values)):
            wma_vals[i] = np.dot(values[i - window + 1:i + 1], weights) / weights.sum()
        return wma_vals
    
    wma_half = wma(close_1d, half_length)
    wma_full = wma(close_1d, 21)
    hma_21 = 2 * wma_half - wma_full
    hma_21 = wma(hma_21, sqrt_length)
    
    # Align 1d HMA to 4h timeframe
    hma_aligned = align_htf_to_ltf(prices, df_1d, hma_21)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        vol_r = vol_ratio[i]
        hma_val = hma_aligned[i]
        
        if np.isnan(vol_r) or np.isnan(hma_val):
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
            if price < exit_low:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            if price > exit_high:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if price > highest and vol_r > 1.8 and hma_val > hma_aligned[i-1]:
                position = 1
                signals[i] = 0.25
            elif price < lowest and vol_r > 1.8 and hma_val < hma_aligned[i-1]:
                position = -1
                signals[i] = -0.25
    
    return signals
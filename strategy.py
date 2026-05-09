#!/usr/bin/env python3
# 6h_Donchian20_1dWeeklyPivot_Breakout_Trend
# Hypothesis: Donchian(20) breakout on 6h with weekly pivot direction filter and volume confirmation.
# Weekly pivot (from weekly high/low/close) determines trend: price above weekly pivot = bullish bias (long only), below = bearish bias (short only).
# Breakout occurs when price breaks Donchian(20) high/low with volume > 1.5x 20-period average.
# Works in bull/bear: weekly pivot filter ensures trend alignment, reducing counter-trend trades.
# Target: 50-150 total trades over 4 years (~12-37/year) to minimize fee drag.

name = "6h_Donchian20_1dWeeklyPivot_Breakout_Trend"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's values for pivot calculation
    ph = np.concatenate([[high_1w[0]], high_1w[:-1]])  # previous weekly high
    pl = np.concatenate([[low_1w[0]], low_1w[:-1]])   # previous weekly low
    pc = np.concatenate([[close_1w[0]], close_1w[:-1]]) # previous weekly close
    
    # Calculate weekly pivot point and support/resistance
    pivot = (ph + pl + pc) / 3.0
    r1 = 2 * pivot - pl
    s1 = 2 * pivot - ph
    r2 = pivot + (ph - pl)
    s2 = pivot - (ph - pl)
    r3 = ph + 2 * (pivot - pl)
    s3 = pl - 2 * (ph - pivot)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Calculate Donchian(20) channels on 6h
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    if len(high) >= 20:
        for i in range(19, len(high)):
            highest_high[i] = np.max(high[i-19:i+1])
            lowest_low[i] = np.min(low[i-19:i+1])
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma[i] = np.mean(volume[i-19:i+1])
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(19, 19)  # Donchian and volume MA need 20 periods
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Determine trend bias from weekly pivot
            bullish_bias = close[i] > pivot_aligned[i]
            bearish_bias = close[i] < pivot_aligned[i]
            
            # Enter long: price breaks above Donchian high AND bullish bias AND volume spike
            if (close[i] > highest_high[i] and 
                bullish_bias and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low AND bearish bias AND volume spike
            elif (close[i] < lowest_low[i] and 
                  bearish_bias and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low OR trend turns bearish (price < weekly pivot)
            if close[i] < lowest_low[i] or close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high OR trend turns bullish (price > weekly pivot)
            if close[i] > highest_high[i] or close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
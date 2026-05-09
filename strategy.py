#!/usr/bin/env python3
# 6h_WeeklyPivot_DonchianBreakout_VolumeFilter
# Hypothesis: Weekly pivot levels (PP, R1, S1) define key weekly support/resistance. 
# Donchian breakout above weekly R1 or below weekly S1 captures institutional breakouts. 
# Volume filter (current > 1.5x 20-bar average) avoids false breakouts. 
# Works in bull/bear: Breakouts capture momentum; weekly pivot context avoids counter-trend trades in ranging weeks.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.

name = "6h_WeeklyPivot_DonchianBreakout_VolumeFilter"
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
    
    # Calculate weekly pivot points from previous week
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's values for pivot calculation
    ph = np.concatenate([[high_1w[0]], high_1w[:-1]])  # previous week high
    pl = np.concatenate([[low_1w[0]], low_1w[:-1]])   # previous week low
    pc = np.concatenate([[close_1w[0]], close_1w[:-1]]) # previous week close
    
    # Weekly pivot point and support/resistance levels
    pp = (ph + pl + pc) / 3.0
    r1 = 2 * pp - pl
    s1 = 2 * pp - ph
    r2 = pp + (ph - pl)
    s2 = pp - (ph - pl)
    
    # Align weekly pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Donchian channel (20-period) for breakout confirmation
    lookback = 20
    highest = np.full_like(high, np.nan)
    lowest = np.full_like(low, np.nan)
    
    for i in range(lookback - 1, len(high)):
        highest[i] = np.max(high[i - lookback + 1:i + 1])
        lowest[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Donchian and volume MA need 20 periods
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(highest[i]) or np.isnan(lowest[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high AND above weekly R1 AND volume spike
            if (close[i] > highest[i] and 
                close[i] > r1_aligned[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low AND below weekly S1 AND volume spike
            elif (close[i] < lowest[i] and 
                  close[i] < s1_aligned[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low OR below weekly S1
            if close[i] < lowest[i] or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high OR above weekly R1
            if close[i] > highest[i] or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
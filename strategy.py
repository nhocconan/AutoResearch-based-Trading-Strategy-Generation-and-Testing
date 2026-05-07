#!/usr/bin/env python3
# 6H_Donchian20_WeeklyPivotTrend_VolumeConfirm
# Hypothesis: Combine Donchian(20) breakout with weekly pivot point trend filter and volume confirmation on 6h timeframe.
# Weekly pivot provides structural support/resistance from higher timeframe (1w), reducing false breakouts.
# Volume confirmation ensures breakouts have institutional participation.
# Works in bull markets (breakouts continue) and bear markets (failed breakouts reverse quickly).
# Target: 20-40 trades/year to stay under 300 total trades over 4 years.

name = "6H_Donchian20_WeeklyPivotTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot point calculation (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly pivot point and support/resistance levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Standard pivot point calculation: (H + L + C) / 3
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    # Resistance 1: (2 * PP) - Low
    r1_1w = (2 * pp_1w) - low_1w
    # Support 1: (2 * PP) - High
    s1_1w = (2 * pp_1w) - high_1w
    
    # Align weekly levels to 6h timeframe (use previous week's levels)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Donchian channel (20-period) on 6h data
    high_max = np.full(n, np.nan)
    low_min = np.full(n, np.nan)
    for i in range(20, n):
        high_max[i] = np.max(high[i-20:i])
        low_min[i] = np.min(low[i-20:i])
    
    # Volume spike detection: 2.0x average volume (50-period for stability)
    vol_ma = np.full(n, np.nan)
    for i in range(50, n):
        vol_ma[i] = np.mean(volume[i-50:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure we have volume MA and Donchian data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(pp_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high, above weekly pivot (bullish bias), volume spike
            if (close[i] > high_max[i] and 
                close[i] > pp_1w_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, below weekly pivot (bearish bias), volume spike
            elif (close[i] < low_min[i] and 
                  close[i] < pp_1w_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to or below weekly support 1
            if close[i] <= s1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to or above weekly resistance 1
            if close[i] >= r1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
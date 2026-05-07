#!/usr/bin/env python3
# 6h_Donchian_20_WeeklyPivotDir_Volume
# Hypothesis: 6h chart strategy using Donchian(20) breakouts filtered by weekly pivot direction from 1w timeframe.
# Long when price breaks above Donchian upper band and weekly pivot is bullish (price > weekly pivot).
# Short when price breaks below Donchian lower band and weekly pivot is bearish (price < weekly pivot).
# Volume confirmation (1.5x average) reduces false breakouts. Weekly pivot provides structural bias
# that works in both bull and bear markets by aligning with higher timeframe trend.
# Target: 15-35 trades/year per symbol to minimize fee drag while maintaining edge.

timeframe = "6h"
name = "6h_Donchian_20_WeeklyPivotDir_Volume"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot direction (higher timeframe than 1d)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly pivot point: (H + L + C) / 3
    weekly_pivot = (df_1w['high'].values + df_1w['low'].values + df_1w['close'].values) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate Donchian channels (20-period) on 6h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection: 1.5x average volume (4-period = 1 day on 6h chart)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 4)  # Ensure we have Donchian and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with volume, and weekly pivot is bullish (price > pivot)
            if (high[i] > donchian_high[i] and 
                volume[i] > 1.5 * vol_ma[i] and 
                close[i] > weekly_pivot_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume, and weekly pivot is bearish (price < pivot)
            elif (low[i] < donchian_low[i] and 
                  volume[i] > 1.5 * vol_ma[i] and 
                  close[i] < weekly_pivot_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Donchian low (reversal signal)
            if low[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Donchian high (reversal signal)
            if high[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
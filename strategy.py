#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
- Uses 6h Donchian(20) channels for breakout signals
- Weekly pivot levels (from 1w) define longer-term structure: long when above weekly pivot, short when below
- Volume confirmation (> 2.0x 24-period average) ensures breakout conviction
- Designed for 6h timeframe targeting 12-25 trades/year (50-100 over 4 years)
- Weekly pivot filter provides structural bias that works in both bull and bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w data for weekly pivot
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly pivot: P = (H + L + C) / 3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate 6h Donchian(20) channels
    donchian_period = 20
    high_max = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    low_min = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume confirmation: > 2.0x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_period, 24)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 6h Donchian upper band AND above weekly pivot AND volume spike
            if (close[i] > high_max[i] and 
                close[i] > weekly_pivot_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 6h Donchian lower band AND below weekly pivot AND volume spike
            elif (close[i] < low_min[i] and 
                  close[i] < weekly_pivot_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian band OR crosses weekly pivot
            exit_signal = False
            
            if position == 1:
                # Exit long when price < 6h Donchian lower band OR < weekly pivot
                if close[i] < low_min[i] or close[i] < weekly_pivot_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when price > 6h Donchian upper band OR > weekly pivot
                if close[i] > high_max[i] or close[i] > weekly_pivot_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_VolumeSpike"
timeframe = "6h"
leverage = 1.0
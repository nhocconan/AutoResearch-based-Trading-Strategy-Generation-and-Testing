#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotTrend_VolumeConfirm
Hypothesis: 6h Donchian(20) breakout traded only in direction of weekly Camarilla pivot trend (price above/below weekly pivot) with volume confirmation (>1.8x 20-bar average). Weekly pivot acts as regime filter: long only when price > weekly pivot, short only when price < weekly pivot. Designed for 12-25 trades/year (~50-100 total over 4 years) to avoid fee drag. Works in both bull/bear via trend-following structure and pivot-based regime filter.
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
    
    # Get weekly data for pivot trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Camarilla pivot (using prior weekly bar)
    weekly_pivot = (close_1w + high_1w + low_1w) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Get daily data for Donchian calculation (more stable than 6h for breakout bands)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian(20) on daily: 20-day high/low
    # Using rolling window on daily data, then align to 6h
    highest_20d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    highest_20d_aligned = align_htf_to_ltf(prices, df_1d, highest_20d)
    lowest_20d_aligned = align_htf_to_ltf(prices, df_1d, lowest_20d)
    
    # Volume confirmation: 1.8x 20-bar average volume on 6h
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian and volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(highest_20d_aligned[i]) or 
            np.isnan(lowest_20d_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Volume confirmation
            volume_confirm = volume[i] > 1.8 * vol_ma20[i]
            
            # Long: price breaks above 20-day high AND price > weekly pivot (uptrend regime) AND volume spike
            long_signal = (close[i] > highest_20d_aligned[i]) and (close[i] > weekly_pivot_aligned[i]) and volume_confirm
            
            # Short: price breaks below 20-day low AND price < weekly pivot (downtrend regime) AND volume spike
            short_signal = (close[i] < lowest_20d_aligned[i]) and (close[i] < weekly_pivot_aligned[i]) and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below weekly pivot (trend regime change)
            exit_signal = close[i] < weekly_pivot_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above weekly pivot (trend regime change)
            exit_signal = close[i] > weekly_pivot_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0
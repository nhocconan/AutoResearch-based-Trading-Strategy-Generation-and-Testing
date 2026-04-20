#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian Breakout with Weekly Pivot Direction and Volume Confirmation
# - Donchian(20) breakout on 6h for entry signals
# - Weekly pivot direction from 1w data as trend filter (only long when price > weekly pivot, short when < weekly pivot)
# - Volume confirmation: current volume > 1.5x 20-period average volume
# - Weekly pivot provides institutional reference points that work in both bull and bear markets
# - Donchian breakouts capture momentum while weekly pivot filters counter-trend moves
# - Volume confirmation ensures breakouts have participation
# - Designed for 6h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot point (standard formula)
    # Pivot = (High + Low + Close) / 3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # 6h Donchian channel (20-period)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Upper band: 20-period high
    donchian_upper = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donchian_lower = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_threshold = avg_volume * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in indicators
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or \
           np.isnan(weekly_pivot_6h[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_ok = volume_6h[i] > volume_threshold[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian upper + above weekly pivot + volume
            if close_6h[i] > donchian_upper[i] and close_6h[i] > weekly_pivot_6h[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower + below weekly pivot + volume
            elif close_6h[i] < donchian_lower[i] and close_6h[i] < weekly_pivot_6h[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian lower or falls below weekly pivot
            if close_6h[i] < donchian_lower[i] or close_6h[i] < weekly_pivot_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian upper or rises above weekly pivot
            if close_6h[i] > donchian_upper[i] or close_6h[i] > weekly_pivot_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0
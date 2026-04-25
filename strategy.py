#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivotDirection_VolumeBreakout
Hypothesis: 6h Donchian(20) breakout in direction of weekly pivot trend (price above/below weekly central pivot) with volume confirmation (>1.5x 20-period average). Uses weekly pivot for longer-term trend filter to avoid counter-trend trades in both bull and bear markets. Targets 12-25 trades/year on 6h timeframe to minimize fee drag while capturing strong directional moves.
"""

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
    
    # Calculate weekly pivot levels (using weekly data as HTF)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot: P = (H + L + C)/3, R1 = 2*P - L, S1 = 2*P - H
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_r1 = 2 * weekly_pivot - low_1w
    weekly_s1 = 2 * weekly_pivot - high_1w
    
    # Align weekly pivot to 6h timeframe (completed weekly bar only)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # 6h Donchian(20) breakout levels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for weekly pivot, Donchian(20), volume MA
    start_idx = max(20, 20)  # Donchian and volume MA both need 20 periods
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high AND above weekly pivot AND volume spike
            long_setup = (close[i] > donchian_high[i]) and \
                         (close[i] > weekly_pivot_aligned[i]) and \
                         volume_spike[i]
            # Short: price breaks below Donchian low AND below weekly pivot AND volume spike
            short_setup = (close[i] < donchian_low[i]) and \
                          (close[i] < weekly_pivot_aligned[i]) and \
                          volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price closes below Donchian low OR below weekly pivot
            if (close[i] < donchian_low[i]) or \
               (close[i] < weekly_pivot_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price closes above Donchian high OR above weekly pivot
            if (close[i] > donchian_high[i]) or \
               (close[i] > weekly_pivot_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_WeeklyPivotDirection_VolumeBreakout"
timeframe = "6h"
leverage = 1.0
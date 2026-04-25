#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1wPivotDirection_VolumeConfirm
Hypothesis: Trade 6h Donchian(20) breakouts aligned with 1-week pivot direction (price above/below weekly pivot) with volume confirmation (>1.5x 20-bar MA). 
Weekly pivot provides structural bias: long when price > weekly pivot, short when price < weekly pivot. 
Donchian breakouts capture momentum in direction of weekly bias. Volume filter avoids false breakouts.
Designed for BTC/ETH: weekly pivot adapts to long-term trend, Donchian provides clear entry/exit, volume confirms validity.
Target: 12-30 trades/year per symbol (50-120 total over 4 years) to avoid fee drag.
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
    
    # Get 1w data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    # Align weekly pivot to 6h timeframe (completed weekly bar only)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate Donchian channels (20-period) on 6h data
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_ma_20
    donchian_lower = low_ma_20
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20) and volume MA (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper AND price > weekly pivot AND volume confirm
            long_setup = (close[i] > donchian_upper[i]) and \
                         (close[i] > weekly_pivot_aligned[i]) and \
                         volume_confirm[i]
            # Short: price breaks below Donchian lower AND price < weekly pivot AND volume confirm
            short_setup = (close[i] < donchian_lower[i]) and \
                          (close[i] < weekly_pivot_aligned[i]) and \
                          volume_confirm[i]
            
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
            # Exit: price re-enters Donchian channel OR price crosses below weekly pivot
            if (close[i] < donchian_upper[i] and close[i] > donchian_lower[i]) or \
               (close[i] < weekly_pivot_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters Donchian channel OR price crosses above weekly pivot
            if (close[i] < donchian_upper[i] and close[i] > donchian_lower[i]) or \
               (close[i] > weekly_pivot_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_1wPivotDirection_VolumeConfirm"
timeframe = "6h"
leverage = 1.0
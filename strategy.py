#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian20_Breakout_VolumeSpike
Hypothesis: On 6h timeframe, Donchian(20) breakouts aligned with weekly pivot bias (above/below weekly pivot) and volume confirmation (>2.0x 20-bar avg) captures institutional moves with low trade frequency. Weekly pivot provides structural bias from higher timeframe (1w), Donchian breakout captures momentum, volume spike confirms participation. Designed for 12-30 trades/year to minimize fee drag. Works in both bull and bear markets via pivot bias filter.
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
    
    # Get 1w data for weekly pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot (standard: (H+L+C)/3) from previous completed week
    prev_high_1w = np.concatenate([[np.nan], high_1w[:-1]])
    prev_low_1w = np.concatenate([[np.nan], low_1w[:-1]])
    prev_close_1w = np.concatenate([[np.nan], close_1w[:-1]])
    
    weekly_pivot = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate Donchian(20) on 6h
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(20, 20)  # Donchian, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(high_ma[i]) or 
            np.isnan(low_ma[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        pivot_val = weekly_pivot_aligned[i]
        upper_donchian = high_ma[i]
        lower_donchian = low_ma[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current volume > 2.0x 20-period average
        volume_spike = vol_val > 2.0 * vol_ma_val
        
        if position == 0:
            # Look for entry signals: Donchian breakout with weekly pivot bias and volume
            # Long: price breaks above upper Donchian with price > weekly pivot and volume spike
            long_signal = (high_val > upper_donchian) and (close_val > pivot_val) and volume_spike
            # Short: price breaks below lower Donchian with price < weekly pivot and volume spike
            short_signal = (low_val < lower_donchian) and (close_val < pivot_val) and volume_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks below lower Donchian (exit long)
            if close_val < lower_donchian:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks above upper Donchian (exit short)
            if close_val > upper_donchian:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "6h_WeeklyPivot_Donchian20_Breakout_VolumeSpike"
timeframe = "6h"
leverage = 1.0
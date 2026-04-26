#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeSpike
Hypothesis: On 6h timeframe, Donchian(20) breakouts in the direction of the weekly pivot trend (price above/below weekly pivot) with volume confirmation (>2.0x 20-bar avg) capture institutional moves. Weekly pivot provides structural bias that works in both bull/bear regimes. Targets 12-30 trades/year to minimize fee drag while maintaining edge via trend alignment and volume filter.
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
    
    # Get 6h data for Donchian calculation (aligned with primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Donchian channels (20-period) on 6h
    highest_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe (already aligned, but ensure proper shifting)
    highest_20_aligned = align_htf_to_ltf(prices, df_6h, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_6h, lowest_20)
    
    # Get weekly data for pivot point calculation (more stable bias)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot point: (H + L + C) / 3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(20, 20)  # Donchian, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_20_aligned[i]) or 
            np.isnan(lowest_20_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or 
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
        highest_20_val = highest_20_aligned[i]
        lowest_20_val = lowest_20_aligned[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current volume > 2.0x 20-period average
        volume_spike = vol_val > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: price breaks above Donchian high with price > weekly pivot (bullish bias) and volume spike
            long_signal = (high_val > highest_20_val) and (close_val > weekly_pivot_val) and volume_spike
            # Short: price breaks below Donchian low with price < weekly pivot (bearish bias) and volume spike
            short_signal = (low_val < lowest_20_val) and (close_val < weekly_pivot_val) and volume_spike
            
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
            # 1. Opposite breakout: price breaks below Donchian low (exit long)
            if close_val < lowest_20_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Loss of bullish bias: price crosses below weekly pivot (exit long)
            elif close_val < weekly_pivot_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks above Donchian high (exit short)
            if close_val > highest_20_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Loss of bearish bias: price crosses above weekly pivot (exit short)
            elif close_val > weekly_pivot_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeSpike"
timeframe = "6h"
leverage = 1.0
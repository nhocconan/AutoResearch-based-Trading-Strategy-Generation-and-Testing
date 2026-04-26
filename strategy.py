#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian20_Breakout_VolumeSpike
Hypothesis: Trade 6h Donchian(20) breakouts only when aligned with weekly pivot direction (price above/below weekly pivot) and volume spike (>90th percentile). Weekly pivot provides institutional reference; Donchian breakout captures momentum; volume spike confirms participation. Only trade in direction of weekly bias to avoid counter-trend whipsaws. Target: 15-25 trades/year per symbol. Uses discrete size 0.25 to limit fees.
"""

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
    
    # Load weekly data ONCE before loop for pivot and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly OHLC for pivot calculation (using prior completed weekly bar)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point: (H + L + C) / 3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Donchian(20) on 6h for breakout
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume spike: volume > 90th percentile of 100-period lookback
    vol_series = pd.Series(volume)
    vol_percentile_90 = vol_series.rolling(window=100, min_periods=100).quantile(0.90).values
    volume_spike = volume > vol_percentile_90
    
    # Fixed position size to control trade frequency (discrete level)
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (100 for volume percentile, 20 for Donchian)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_percentile_90[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        pivot_val = weekly_pivot_aligned[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        # Entry conditions: Donchian breakout with volume spike AND aligned with weekly pivot
        long_entry = (close_val > donch_high) and vol_spike and (close_val > pivot_val)
        short_entry = (close_val < donch_low) and vol_spike and (close_val < pivot_val)
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price retreats below weekly pivot (trend change)
            if close_val < pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price rises above weekly pivot (trend change)
            if close_val > pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyPivot_Donchian20_Breakout_VolumeSpike"
timeframe = "6h"
leverage = 1.0
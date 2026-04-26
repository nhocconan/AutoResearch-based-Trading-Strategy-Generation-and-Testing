#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian20_Breakout_VolumeSpike_v1
Hypothesis: Trade 6h Donchian(20) breakouts in direction of weekly Camarilla pivot bias (based on weekly close vs Pivot) with volume confirmation (2.0x median). Weekly pivot provides structural bias that works in bull/bear markets. Volume spike confirms institutional participation. Target: 12-37 trades/year on 6h timeframe.
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
    
    # Get weekly data for pivot bias calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (using prior week OHLC)
    prev_week_close = df_1w['close'].shift(1).values
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    # Weekly bias: 1 if close > pivot (bullish), -1 if close < pivot (bearish)
    weekly_bias = np.where(prev_week_close > weekly_pivot, 1, -1)
    
    # Get daily data for Donchian channel calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Donchian(20) from daily high/low
    donchian_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Align HTF indicators to 6h timeframe
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume confirmation: 2.0x median volume (20-period) for signal
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of weekly bias (needs 1), Donchian (20), volume median (20)
    start_idx = max(1, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_bias_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_median[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        bias_val = weekly_bias_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        
        if position == 0:
            # Long: break above Donchian high with volume spike, and weekly bullish bias
            long_signal = (high_val > donchian_high_val) and \
                          (volume_val > 2.0 * vol_median_val) and \
                          (bias_val == 1)
            
            # Short: break below Donchian low with volume spike, and weekly bearish bias
            short_signal = (low_val < donchian_low_val) and \
                           (volume_val > 2.0 * vol_median_val) and \
                           (bias_val == -1)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price retraces to midpoint of Donchian channel
            midpoint = (donchian_high_val + donchian_low_val) / 2.0
            if close_val <= midpoint:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price retraces to midpoint of Donchian channel
            midpoint = (donchian_high_val + donchian_low_val) / 2.0
            if close_val >= midpoint:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_Donchian20_Breakout_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0
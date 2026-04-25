#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation
Hypothesis: Weekly pivot (from 1w OHLC) provides strong structural support/resistance.
Donchian breakout in direction of weekly pivot bias with volume confirmation captures
strong momentum moves. Works in bull/bear by following weekly structural bias.
Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w pivot points (using prior week OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Prior week OHLC for pivot calculation (shifted by 1 to avoid look-ahead)
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    # Standard pivot point calculation
    pivot_point = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    r1 = 2 * pivot_point - prev_week_low
    s1 = 2 * pivot_point - prev_week_high
    r2 = pivot_point + (prev_week_high - prev_week_low)
    s2 = pivot_point - (prev_week_high - prev_week_low)
    
    # Align weekly pivot levels to 6h timeframe (waits for 1w bar close)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_point)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Calculate 6h Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # Need 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        pivot_level = pivot_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        r2_level = r2_aligned[i]
        s2_level = s2_aligned[i]
        donchian_high_level = donchian_high[i]
        donchian_low_level = donchian_low[i]
        
        # Volume spike: current volume > 2.0 * 20-period average volume
        volume_spike = curr_volume > 2.0 * vol_ma_20[i]
        
        # Donchian breakout conditions
        broke_above_donchian = curr_close > donchian_high_level
        broke_below_donchian = curr_close < donchian_low_level
        
        # Weekly pivot bias conditions
        above_pivot = curr_close > pivot_level
        below_pivot = curr_close < pivot_level
        above_r1 = curr_close > r1_level
        below_s1 = curr_close < s1_level
        
        # Exit conditions: opposite Donchian break
        if position != 0:
            # Exit long: close breaks below Donchian low
            if position == 1:
                if curr_close < donchian_low_level:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: close breaks above Donchian high
            elif position == -1:
                if curr_close > donchian_high_level:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with weekly pivot bias and volume
        if position == 0:
            # Long: break above Donchian high AND above weekly pivot AND volume spike
            long_condition = broke_above_donchian and above_pivot and volume_spike
            
            # Short: break below Donchian low AND below weekly pivot AND volume spike
            short_condition = broke_below_donchian and below_pivot and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_1wPivot_Direction_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0
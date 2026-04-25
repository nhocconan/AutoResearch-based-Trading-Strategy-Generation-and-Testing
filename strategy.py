#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Spike Confirmation
Hypothesis: 6h Donchian breakouts capture medium-term momentum. Weekly pivot (from 1w data) 
provides major trend filter to avoid counter-trend trades. Volume confirmation ensures 
breakout strength. Designed for BTC/ETH with 50-150 total trades over 4 years (12-37/year) 
to balance opportunity and fee drag. Works in bull markets (breakouts above weekly pivot) 
and bear markets (breakouts below weekly pivot) by using weekly pivot as trend filter.
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
    
    # Get weekly data for pivot calculation (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:  # Need at least 2 weeks for pivot calculation
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    # Using standard pivot: P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # We'll use R1/S1 as breakout levels, pivot as trend filter
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's OHLC (shifted by 1 to avoid look-ahead)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    
    # First week has no previous week
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    prev_close_1w[0] = np.nan
    
    # Calculate weekly pivot and levels
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    r1_1w = 2 * pivot_1w - prev_low_1w  # Resistance 1
    s1_1w = 2 * pivot_1w - prev_high_1w  # Support 1
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Calculate 6h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Calculate 20-period volume MA for volume spike confirmation (6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian, volume MA, and weekly pivot
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Trend filter: price relative to weekly pivot
        # In bull mode (price > pivot): look for longs on Donchian breakouts
        # In bear mode (price < pivot): look for shorts on Donchian breakdowns
        bull_mode = curr_close > pivot_val
        bear_mode = curr_close < pivot_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for breakout signals
            # Long: price breaks above Donchian high with volume confirmation in bull mode
            long_breakout = (curr_close > donchian_high_val) and volume_confirm and bull_mode
            # Short: price breaks below Donchian low with volume confirmation in bear mode
            short_breakout = (curr_close < donchian_low_val) and volume_confirm and bear_mode
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                position = 0
        elif position == 1:
            # Exit long: price closes below Donchian low OR weekly pivot breaks down
            if curr_close < donchian_low_val or curr_close < pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian high OR weekly pivot breaks up
            if curr_close > donchian_high_val or curr_close > pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_Direction_VolumeSpike"
timeframe = "6h"
leverage = 1.0
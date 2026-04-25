#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Spike
Hypothesis: Donchian breakouts capture momentum, filtered by weekly pivot 
trend (price above/below weekly pivot) for directional bias, with volume 
confirmation to avoid false breakouts. Works in bull/bear by trading both 
long and short breakouts with trend filter. Designed for 6h timeframe with 
50-150 total trades over 4 years.
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
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard: P = (H+L+C)/3, R1 = 2P-L, S1 = 2P-H)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    
    # Align weekly pivot to 6h (completed weekly bar only)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Calculate 20-period Donchian channels (6h)
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
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
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
        pivot_val = pivot_1w_aligned[i]
        r1_val = r1_1w_aligned[i]
        s1_val = s1_1w_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Trend filter: price relative to weekly pivot (directional bias)
        # Above pivot = bullish bias (favor longs), below pivot = bearish bias (favor shorts)
        bullish_bias = curr_close > pivot_val
        bearish_bias = curr_close < pivot_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for breakout signals with trend and volume confirmation
            # Long: price breaks above Donchian high with volume confirmation in bullish bias
            long_breakout = (curr_high > donchian_high_val) and volume_confirm and bullish_bias
            # Short: price breaks below Donchian low with volume confirmation in bearish bias
            short_breakout = (curr_low < donchian_low_val) and volume_confirm and bearish_bias
            
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

name = "6h_Donchian_Breakout_WeeklyPivot_Direction_VolumeSpike"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
"""
6h Donchian(20) breakout + 1d Weekly Pivot R4/S4 direction filter + Volume confirmation
Hypothesis: Donchian breakouts capture momentum, while weekly pivot R4/S4 levels act as
major support/resistance from institutional order flow. Breakouts aligned with weekly
pivot direction have higher follow-through. Volume confirms real participation.
Works in bull/bear via breakout logic (long on upside breakouts, short on downside).
Target: 12-37 trades/year on 6h.
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
    
    # Load 1d data ONCE before loop for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior week (Mon-Fri)
    # Need at least 5 daily bars for a week
    weekly_high = pd.Series(df_1d['high'].values).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(df_1d['low'].values).rolling(window=5, min_periods=5).min().values
    weekly_close = df_1d['close'].values
    
    # Weekly Camarilla-like pivots (R4/S4 are most significant)
    # R4 = weekly_close + (weekly_high - weekly_low) * 1.1
    # S4 = weekly_close - (weekly_high - weekly_low) * 1.1
    weekly_r4 = weekly_close + (weekly_high - weekly_low) * 1.1
    weekly_s4 = weekly_close - (weekly_high - weekly_low) * 1.1
    
    # Align weekly levels to 6h timeframe (use prior week's levels)
    weekly_r4_aligned = align_htf_to_ltf(prices, df_1d, weekly_r4)
    weekly_s4_aligned = align_htf_to_ltf(prices, df_1d, weekly_s4)
    
    # Donchian channel (20-period) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.8 * 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(20, 30, 5) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_r4_aligned[i]) or np.isnan(weekly_s4_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Donchian breakout conditions
        breakout_long = curr_high > donchian_high[i-1]  # Break above prior period's high
        breakout_short = curr_low < donchian_low[i-1]   # Break below prior period's low
        
        # Weekly pivot direction filter
        # Long bias: price above weekly S4 (bullish bias)
        # Short bias: price below weekly R4 (bearish bias)
        long_bias = curr_close > weekly_s4_aligned[i]
        short_bias = curr_close < weekly_r4_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: Donchian breakout + weekly bias + volume spike
            long_entry = breakout_long and long_bias and vol_spike
            short_entry = breakout_short and short_bias and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price retouches Donchian low or weekly S4
            if curr_low <= donchian_low[i] or curr_close <= weekly_s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price retouches Donchian high or weekly R4
            if curr_high >= donchian_high[i] or curr_close >= weekly_r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotR4S4_VolumeSpike"
timeframe = "6h"
leverage = 1.0
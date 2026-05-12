#!/usr/bin/env python3
name = "6h_Donchian_Breakout_WeeklyPivot_Direction"
timeframe = "6h"
leverage = 1.0

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
    
    # Load daily data once for weekly pivot and Donchian
    df_1d = get_htf_data(prices, '1d')
    
    # Weekly pivot from previous week (Monday open, Friday close)
    high_w = df_1d['high'].values
    low_w = df_1d['low'].values
    close_w = df_1d['close'].values
    
    # Calculate weekly pivot (previous week)
    p_w = (high_w + low_w + close_w) / 3
    # Resistance 1 and Support 1 from weekly pivot
    r1_w = p_w + (high_w - low_w) * 1.1 / 4
    s1_w = p_w - (high_w - low_w) * 1.1 / 4
    
    # Align weekly pivot to 6h (wait for weekly close)
    r1_w_aligned = align_htf_to_ltf(prices, df_1d, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_1d, s1_w)
    
    # Donchian channel (20-period) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # ensure Donchian has enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_w_aligned[i]) or np.isnan(s1_w_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + above weekly R1 + volume confirmation
            if (close[i] > donchian_high[i] and 
                close[i] > r1_w_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + below weekly S1 + volume confirmation
            elif (close[i] < donchian_low[i] and 
                  close[i] < s1_w_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below weekly S1
            if close[i] < s1_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above weekly R1
            if close[i] > r1_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
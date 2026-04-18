#!/usr/bin/env python3
"""
4h_WideRange_Breakout_Volume_Trend
Hypothesis: Uses 12h wide range (high-low) filter to identify expansion periods, then trades breakouts
of 4h Donchian channels with volume confirmation. Works in both bull (breakouts up) and bear (breakouts down)
by capturing volatility expansion phases. Targets 20-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for range filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h range (high-low) and its 20-period average
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    range_12h = high_12h - low_12h
    
    range_ma = np.full(len(range_12h), np.nan)
    for i in range(20, len(range_12h)):
        range_ma[i] = np.mean(range_12h[i-20:i])
    
    # Wide range condition: current range > 1.5 x 20-period average
    wide_range = range_12h > (range_ma * 1.5)
    
    # Get 4h data for Donchian channels (20-period)
    high_4h = high
    low_4h = low
    
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    
    for i in range(20, n):
        upper_channel[i] = np.max(high_4h[i-20:i])
        lower_channel[i] = np.min(low_4h[i-20:i])
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    # Align 12h wide range to 4h timeframe
    wide_range_aligned = align_htf_to_ltf(prices, df_12h, wide_range)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(wide_range_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above 4h upper channel with volume spike and wide range
            if (close[i] > upper_channel[i] and vol_spike[i] and wide_range_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below 4h lower channel with volume spike and wide range
            elif (close[i] < lower_channel[i] and vol_spike[i] and wide_range_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below 4h lower channel
            if close[i] < lower_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above 4h upper channel
            if close[i] > upper_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WideRange_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
"""
12h_Weekly_Donchian_Breakout_Volume_Filter
Hypothesis: Uses weekly Donchian breakout (20-period) on 12h timeframe with volume confirmation and 1d EMA trend filter.
Designed for low trade frequency (<30/year) to minimize fee drag, works in both bull and bear markets by following higher timeframe structure.
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
    
    # Get weekly data for Donchian breakout signals
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian channels (20-period)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    upper_channel = np.full(len(high_weekly), np.nan)
    lower_channel = np.full(len(low_weekly), np.nan)
    
    for i in range(20, len(high_weekly)):
        upper_channel[i] = np.max(high_weekly[i-20:i])
        lower_channel[i] = np.min(low_weekly[i-20:i])
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema34[33] = np.mean(close_1d[0:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34[i] = close_1d[i] * alpha + ema34[i-1] * (1 - alpha)
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    # Align weekly channels to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_weekly, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_weekly, lower_channel)
    
    # Align 1d EMA to 12h timeframe
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20, 34)
    
    for i in range(start_idx, n):
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above weekly upper channel with volume spike and 1d uptrend
            if (close[i] > upper_aligned[i] and vol_spike[i] and 
                close[i] > ema34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below weekly lower channel with volume spike and 1d downtrend
            elif (close[i] < lower_aligned[i] and vol_spike[i] and 
                  close[i] < ema34_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below weekly lower channel or 1d trend turns down
            if (close[i] < lower_aligned[i] or close[i] < ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above weekly upper channel or 1d trend turns up
            if (close[i] > upper_aligned[i] or close[i] > ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Weekly_Donchian_Breakout_Volume_Filter"
timeframe = "12h"
leverage = 1.0
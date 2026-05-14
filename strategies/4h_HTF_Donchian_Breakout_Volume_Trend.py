#!/usr/bin/env python3
"""
4h_HTF_Donchian_Breakout_Volume_Trend
Hypothesis: Combines 12h Donchian breakout with volume confirmation and 4h EMA trend filter.
Uses higher timeframe structure (12h) for signal direction to reduce false breakouts and
improve performance in both bull and bear markets. Targets 25-40 trades/year.
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
    
    # Get 12h data for Donchian breakout signals
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    upper_channel = np.full(len(high_12h), np.nan)
    lower_channel = np.full(len(low_12h), np.nan)
    
    for i in range(20, len(high_12h)):
        upper_channel[i] = np.max(high_12h[i-20:i])
        lower_channel[i] = np.min(low_12h[i-20:i])
    
    # Get 4h data for EMA trend filter
    ema20 = np.full(n, np.nan)
    if n >= 20:
        ema20[19] = np.mean(close[0:20])
        alpha = 2 / (20 + 1)
        for i in range(20, n):
            ema20[i] = close[i] * alpha + ema20[i-1] * (1 - alpha)
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    # Align 12h channels to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_channel)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema20[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above 12h upper channel with volume spike and 4h uptrend
            if (close[i] > upper_aligned[i] and vol_spike[i] and 
                close[i] > ema20[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below 12h lower channel with volume spike and 4h downtrend
            elif (close[i] < lower_aligned[i] and vol_spike[i] and 
                  close[i] < ema20[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below 12h lower channel or 4h trend turns down
            if (close[i] < lower_aligned[i] or close[i] < ema20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above 12h upper channel or 4h trend turns up
            if (close[i] > upper_aligned[i] or close[i] > ema20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_HTF_Donchian_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0
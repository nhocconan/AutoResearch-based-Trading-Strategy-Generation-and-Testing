#!/usr/bin/env python3
"""
4h_HTF_Donchian_Breakout_Volume_Trend_v2
Hypothesis: Uses 1d Donchian(40) breakout with volume confirmation and 4h EMA(34) trend filter.
Higher timeframe structure reduces false breakouts. Targets 20-30 trades/year.
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
    
    # Get 1d data for Donchian breakout signals
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Donchian channels (40-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    upper_channel = np.full(len(high_1d), np.nan)
    lower_channel = np.full(len(low_1d), np.nan)
    
    for i in range(40, len(high_1d)):
        upper_channel[i] = np.max(high_1d[i-40:i])
        lower_channel[i] = np.min(low_1d[i-40:i])
    
    # Get 4h data for EMA trend filter
    ema34 = np.full(n, np.nan)
    if n >= 34:
        ema34[33] = np.mean(close[0:34])
        alpha = 2 / (34 + 1)
        for i in range(34, n):
            ema34[i] = close[i] * alpha + ema34[i-1] * (1 - alpha)
    
    # Volume spike: current volume > 2.5 x 40-period average
    vol_ma = np.full(n, np.nan)
    for i in range(40, n):
        vol_ma[i] = np.mean(volume[i-40:i])
    vol_spike = volume > (vol_ma * 2.5)
    
    # Align 1d channels to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_channel)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 40)
    
    for i in range(start_idx, n):
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema34[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above 1d upper channel with volume spike and 4h uptrend
            if (close[i] > upper_aligned[i] and vol_spike[i] and 
                close[i] > ema34[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below 1d lower channel with volume spike and 4h downtrend
            elif (close[i] < lower_aligned[i] and vol_spike[i] and 
                  close[i] < ema34[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below 1d lower channel or 4h trend turns down
            if (close[i] < lower_aligned[i] or close[i] < ema34[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above 1d upper channel or 4h trend turns up
            if (close[i] > upper_aligned[i] or close[i] > ema34[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_HTF_Donchian_Breakout_Volume_Trend_v2"
timeframe = "4h"
leverage = 1.0
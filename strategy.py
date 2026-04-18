#!/usr/bin/env python3
"""
6h_Donchian_Breakout_WeeklyPivotDirection_Volume
Hypothesis: 6-hour Donchian breakout with weekly pivot direction filter and volume confirmation.
Trades breakouts from 20-period 6-hour high/low only when aligned with weekly pivot trend (price above/below weekly pivot)
and accompanied by volume spike. Designed for low frequency (~15-30 trades/year) with strong performance
in both bull and bear markets by combining intraday breakout structure with weekly trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly pivot
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot (average of weekly high, low, close)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    weekly_pivot = np.full(len(close_1w), np.nan)
    for i in range(len(close_1w)):
        weekly_pivot[i] = (high_1w[i] + low_1w[i] + close_1w[i]) / 3.0
    
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate 6-hour Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above 6h Donchian high with volume spike and above weekly pivot
            if (high[i] > donchian_high[i] and vol_spike[i] and 
                close[i] > weekly_pivot_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below 6h Donchian low with volume spike and below weekly pivot
            elif (low[i] < donchian_low[i] and vol_spike[i] and 
                  close[i] < weekly_pivot_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: break below 6h Donchian low or price crosses below weekly pivot
            if (low[i] < donchian_low[i] or close[i] < weekly_pivot_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above 6h Donchian high or price crosses above weekly pivot
            if (high[i] > donchian_high[i] or close[i] > weekly_pivot_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian_Breakout_WeeklyPivotDirection_Volume"
timeframe = "6h"
leverage = 1.0
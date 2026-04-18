#!/usr/bin/env python3
"""
6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation
Weekly pivot sets directional bias, Donchian breakout on 6h with volume confirmation.
Trades only in direction of higher timeframe weekly pivot bias.
Designed to work in both bull and bear markets by filtering breakouts with weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot (standard pivot point)
    pivot = (high_1w + low_1w + close_1w) / 3.0
    # Bias: above pivot = bullish bias, below pivot = bearish bias
    weekly_bias = pivot  # we'll use this as reference level
    
    # Align weekly pivot to 6h timeframe
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias)
    
    # Donchian channel on 6h (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation (1.5x 10-period average)
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(100, lookback)  # ensure enough history
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_bias_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        bias = weekly_bias_aligned[i]
        upper = highest_high[i]
        lower = lowest_low[i]
        
        if position == 0:
            # Long: break above Donchian high with volume and above weekly pivot
            if (price > upper and 
                volume_confirm[i] and 
                price > bias):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume and below weekly pivot
            elif (price < lower and 
                  volume_confirm[i] and 
                  price < bias):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position: hold until break below Donchian low
            signals[i] = 0.25
            if price < lower:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position: hold until break above Donchian high
            signals[i] = -0.25
            if price > upper:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0
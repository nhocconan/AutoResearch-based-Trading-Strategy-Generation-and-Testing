#!/usr/bin/env python3
"""
6h Donchian Breakout with Weekly Pivot Direction and Volume Confirmation
Hypothesis: Donchian(20) breakouts on 6h timeframe capture momentum. 
Weekly pivot direction (from 1w) filters breakouts to trade only in alignment 
with higher timeframe bias. Volume confirmation ensures institutional participation. 
Works in both bull and bear markets by following breakout direction with 
weekly bias filter reducing whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_highest(arr, period):
    """Calculate rolling highest"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    result = np.full(len(arr), np.nan)
    for i in range(period-1, len(arr)):
        result[i] = np.max(arr[i-period+1:i+1])
    return result

def calculate_lowest(arr, period):
    """Calculate rolling lowest"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    result = np.full(len(arr), np.nan)
    for i in range(period-1, len(arr)):
        result[i] = np.min(arr[i-period+1:i+1])
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points using prior week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot: P = (H + L + C)/3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    # Weekly bias: above pivot = bullish, below = bearish
    weekly_bullish = close_1w > weekly_pivot
    
    # Align weekly bias to 6h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    
    # Donchian channels (20-period)
    donchian_high = calculate_highest(high, 20)
    donchian_low = calculate_lowest(low, 20)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Warmup for Donchian(20) and weekly alignment
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(weekly_bullish_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: breakout above Donchian high with weekly bullish bias and volume
            if (close[i] > donchian_high[i] and 
                weekly_bullish_aligned[i] > 0.5 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian low with weekly bearish bias and volume
            elif (close[i] < donchian_low[i] and 
                  weekly_bullish_aligned[i] < 0.5 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below Donchian low or weekly bias flips bearish
            if close[i] < donchian_low[i] or weekly_bullish_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above Donchian high or weekly bias flips bullish
            if close[i] > donchian_high[i] or weekly_bullish_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0
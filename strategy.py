#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Donchian20_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Donchian channel and weekly pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Donchian(20) on daily: upper/lower bands
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Weekly pivot points (using previous week)
    pivot = np.zeros(len(df_1w))
    R1 = np.zeros(len(df_1w))
    S1 = np.zeros(len(df_1w))
    R2 = np.zeros(len(df_1w))
    S2 = np.zeros(len(df_1w))
    
    for i in range(1, len(df_1w)):
        high_prev = df_1w['high'].iloc[i-1]
        low_prev = df_1w['low'].iloc[i-1]
        close_prev = df_1w['close'].iloc[i-1]
        pp = (high_prev + low_prev + close_prev) / 3.0
        pivot[i] = pp
        R1[i] = 2 * pp - low_prev
        S1[i] = 2 * pp - high_prev
        R2[i] = pp + (high_prev - low_prev)
        S2[i] = pp - (high_prev - low_prev)
    
    # Align weekly pivot to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    R2_aligned = align_htf_to_ltf(prices, df_1w, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1w, S2)
    
    # Volume spike: current volume > 1.8x 30-period average
    vol_ma30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high, price above weekly pivot, volume spike
            long_cond = (close[i] > donch_high_aligned[i] and 
                        close[i] > pivot_aligned[i] and
                        volume_spike[i])
            
            # Short: price breaks below Donchian low, price below weekly pivot, volume spike
            short_cond = (close[i] < donch_low_aligned[i] and 
                         close[i] < pivot_aligned[i] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below Donchian low
            if close[i] < donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above Donchian high
            if close[i] > donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
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
    
    # Get weekly data for Donchian channel (20-period)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channels (20-period)
    upper = np.full_like(high_1w, np.nan)
    lower = np.full_like(low_1w, np.nan)
    
    for i in range(20, len(high_1w)):
        upper[i] = np.max(high_1w[i-20:i])
        lower[i] = np.min(low_1w[i-20:i])
    
    # Get daily data for pivot levels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate weekly pivot points (based on previous week)
    pivot = np.full_like(high_1w, np.nan)
    R1 = np.full_like(high_1w, np.nan)
    S1 = np.full_like(low_1w, np.nan)
    R2 = np.full_like(high_1w, np.nan)
    S2 = np.full_like(low_1w, np.nan)
    
    for i in range(1, len(high_1w)):
        prev_high = high_1w[i-1]
        prev_low = low_1w[i-1]
        prev_close = close_1d[i-1]  # Use daily close for pivot calc
        pp = (prev_high + prev_low + prev_close) / 3.0
        pivot[i] = pp
        R1[i] = 2 * pp - prev_low
        S1[i] = 2 * pp - prev_high
        R2[i] = pp + (prev_high - prev_low)
        S2[i] = pp - (prev_high - prev_low)
    
    # Calculate daily volume average (20-period)
    vol_ma_1d = np.full_like(volume_1d, np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Align all weekly data to 6h timeframe
    upper_6h = align_htf_to_ltf(prices, df_1w, upper)
    lower_6h = align_htf_to_ltf(prices, df_1w, lower)
    pivot_6h = align_htf_to_ltf(prices, df_1w, pivot)
    R1_6h = align_htf_to_ltf(prices, df_1w, R1)
    S1_6h = align_htf_to_ltf(prices, df_1w, S1)
    R2_6h = align_htf_to_ltf(prices, df_1w, R2)
    S2_6h = align_htf_to_ltf(prices, df_1w, S2)
    
    # Align daily volume to 6h timeframe
    vol_ma_1d_6h = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure we have enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_6h[i]) or np.isnan(lower_6h[i]) or 
            np.isnan(pivot_6h[i]) or np.isnan(R1_6h[i]) or np.isnan(S1_6h[i]) or
            np.isnan(R2_6h[i]) or np.isnan(S2_6h[i]) or 
            np.isnan(vol_ma_1d_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x daily 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma_1d_6h[i]
        
        if position == 0:
            # Long: price breaks above weekly upper Donchian AND above weekly R1 pivot
            if close[i] > upper_6h[i] and close[i] > R1_6h[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly lower Donchian AND below weekly S1 pivot
            elif close[i] < lower_6h[i] and close[i] < S1_6h[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below weekly pivot OR below weekly S1
            if close[i] < pivot_6h[i] or close[i] < S1_6h[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above weekly pivot OR above weekly R1
            if close[i] > pivot_6h[i] or close[i] > R1_6h[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian_WeeklyPivot_Breakout_Volume"
timeframe = "6h"
leverage = 1.0
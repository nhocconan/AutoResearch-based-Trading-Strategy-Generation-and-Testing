#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation
Hypothesis: Weekly pivot levels (from 1w) provide strong directional bias on 6h.
Breakouts above weekly R1 or below weekly S1 with Donchian(20) confirmation and
volume spike capture momentum moves. Weekly pivot direction filters trades to
align with higher timeframe structure, working in both bull (long bias) and bear
(short bias) markets. Volume confirmation ensures breakout validity. Targets
50-150 total trades over 4 years to minimize fee drag.
"""

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
    
    # Get weekly data for pivot direction (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points and R1/S1 levels
    # Pivot = (H + L + C) / 3
    # R1 = (2 * Pivot) - L
    # S1 = (2 * Pivot) - H
    weekly_pivot = np.full(len(df_1w), np.nan)
    weekly_r1 = np.full(len(df_1w), np.nan)
    weekly_s1 = np.full(len(df_1w), np.nan)
    
    for i in range(len(df_1w)):
        wk_high = df_1w['high'].iloc[i]
        wk_low = df_1w['low'].iloc[i]
        wk_close = df_1w['close'].iloc[i]
        pivot = (wk_high + wk_low + wk_close) / 3.0
        r1 = (2 * pivot) - wk_low
        s1 = (2 * pivot) - wk_high
        weekly_pivot[i] = pivot
        weekly_r1[i] = r1
        weekly_s1[i] = s1
    
    # Align weekly pivot levels to 6h timeframe
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Get 1d data for Donchian(20) breakout levels (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian(20) channels from 1d
    donch_high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # Calculate 20-period volume MA for volume confirmation (6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for weekly alignment, Donchian, volume MA to propagate
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        weekly_r1 = weekly_r1_aligned[i]
        weekly_s1 = weekly_s1_aligned[i]
        donch_high = donch_high_aligned[i]
        donch_low = donch_low_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 1.8 * 20-period average
        volume_confirm = curr_volume > 1.8 * vol_ma
        
        if position == 0:
            # Long breakout: close above weekly R1 AND above Donchian high with volume confirmation
            long_breakout = (curr_close > weekly_r1) and (curr_close > donch_high) and volume_confirm
            # Short breakdown: close below weekly S1 AND below Donchian low with volume confirmation
            short_breakout = (curr_close < weekly_s1) and (curr_close < donch_low) and volume_confirm
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below weekly S1 (reversal signal)
            if curr_close < weekly_s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above weekly R1 (reversal signal)
            if curr_close > weekly_r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_Direction_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0
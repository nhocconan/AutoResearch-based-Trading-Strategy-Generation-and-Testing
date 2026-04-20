#!/usr/bin/env python3
"""
12h_1d_Camarilla_R1S1_Breakout_VolumeFilter
Hypothesis: Trade daily Camarilla pivot R1/S1 breakouts on 12h timeframe with daily volume confirmation.
Long when 12h price breaks above daily R1 with volume spike; short when breaks below daily S1 with volume spike.
Daily Camarilla levels provide strong daily support/resistance. Volume filter ensures institutional participation.
Works in bull/bear: breaks indicate momentum continuation, volume confirms validity.
Target: 60-120 total trades over 4 years (15-30/year) with position size 0.25.
"""

name = "12h_1d_Camarilla_R1S1_Breakout_VolumeFilter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for Camarilla levels and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily volume average for spike detection (20-period)
    vol_1d = df_1d['volume'].values
    vol_avg_1d = np.full(len(vol_1d), np.nan)
    for i in range(len(vol_1d)):
        if i >= 19:  # 20-period average
            vol_avg_1d[i] = np.mean(vol_1d[i-19:i+1])
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate daily Camarilla levels for each daily bar
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Arrays to store daily R1 and S1 levels
    daily_r1 = np.full_like(daily_close, np.nan)
    daily_s1 = np.full_like(daily_close, np.nan)
    
    # Calculate for each daily bar (starting from index 1 to avoid look-ahead)
    for j in range(1, len(daily_close)):
        range_val = daily_high[j-1] - daily_low[j-1]
        if range_val > 0:
            daily_r1[j] = daily_close[j-1] + (range_val * 1.1 / 12)
            daily_s1[j] = daily_close[j-1] - (range_val * 1.1 / 12)
    
    # Align the daily R1/S1 to 12h timeframe
    daily_r1_aligned = align_htf_to_ltf(prices, df_1d, daily_r1)
    daily_s1_aligned = align_htf_to_ltf(prices, df_1d, daily_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 10  # Need enough data for daily alignment
    
    for i in range(start_idx, n):
        current_close = prices['close'].iloc[i]
        current_volume = prices['volume'].iloc[i]
        
        # Volume spike: current volume > 1.5x daily average volume
        vol_spike = (not np.isnan(vol_avg_1d_aligned[i]) and 
                     current_volume > 1.5 * vol_avg_1d_aligned[i])
        
        if position == 0:
            # Long: price breaks above daily R1 with volume spike
            if (not np.isnan(daily_r1_aligned[i]) and 
                current_close > daily_r1_aligned[i] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily S1 with volume spike
            elif (not np.isnan(daily_s1_aligned[i]) and 
                  current_close < daily_s1_aligned[i] and vol_spike):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below daily S1 or volume dries up
            if ((not np.isnan(daily_s1_aligned[i]) and 
                 current_close < daily_s1_aligned[i]) or
                (not np.isnan(vol_avg_1d_aligned[i]) and 
                 current_volume < 0.5 * vol_avg_1d_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above daily R1 or volume dries up
            if ((not np.isnan(daily_r1_aligned[i]) and 
                 current_close > daily_r1_aligned[i]) or
                (not np.isnan(vol_avg_1d_aligned[i]) and 
                 current_volume < 0.5 * vol_avg_1d_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
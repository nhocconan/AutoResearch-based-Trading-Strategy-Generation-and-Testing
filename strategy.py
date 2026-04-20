#!/usr/bin/env python3
"""
1d_1w_Camarilla_R1S1_Breakout_VolumeFilter
Hypothesis: Trade weekly Camarilla pivot R1/S1 breakouts on daily timeframe with weekly volume confirmation.
Long when daily price breaks above weekly R1 with volume spike; short when breaks below weekly S1 with volume spike.
Weekly Camarilla levels provide strong weekly support/resistance. Volume filter ensures institutional participation.
Works in bull/bear: breaks indicate momentum continuation, volume confirms validity.
Target: 20-50 total trades over 4 years (5-12.5/year) with position size 0.25.
"""

name = "1d_1w_Camarilla_R1S1_Breakout_VolumeFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for Camarilla levels and volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly volume average for spike detection (20-period)
    vol_1w = df_1w['volume'].values
    vol_avg_1w = np.full(len(vol_1w), np.nan)
    for i in range(len(vol_1w)):
        if i >= 19:  # 20-period average
            vol_avg_1w[i] = np.mean(vol_1w[i-19:i+1])
    vol_avg_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 5  # Need enough data for weekly alignment
    
    for i in range(start_idx, n):
        # Skip until we have at least one full week of prior data for Camarilla calculation
        if i < 5:  # Need at least 5 days prior for weekly calculation
            continue
            
        # Calculate weekly Camarilla levels using prior week's OHLC
        # Get index of current week's start in weekly data
        # We need the previous completed weekly bar
        week_idx = 0
        # Find the weekly bar that corresponds to current daily index
        # Since we aligned the weekly data, we can use the same index concept
        # but we need to ensure we're using the previous week's data
        
        # Simpler approach: use the aligned weekly data directly
        # We need to get the weekly OHLC from the prior completed week
        # Since we're on daily timeframe, we look at the aligned weekly values
        # that represent the completed weekly bar
        
        # Get the weekly high, low, close from the aligned data
        # We need to extract these from the original weekly dataframe
        # and align them properly
        
        # For simplicity, we'll calculate weekly levels using the weekly dataframe directly
        # and align the results
        
        # Need at least 2 weekly bars to have a previous week
        if len(df_1w) < 2:
            continue
            
        # Get the previous completed weekly bar
        # We'll use index -2 to ensure we're using fully completed weekly data
        # (avoiding look-ahead)
        if len(df_1w) >= 2:
            prev_week = df_1w.iloc[-2] if len(df_1w) >= 2 else df_1w.iloc[-1]
            # But we need to do this properly for each point in time
            
        # Better approach: calculate weekly levels for each weekly bar, then align
        if len(df_1w) >= 5:
            # Calculate weekly Camarilla levels for each weekly bar
            weekly_high = df_1w['high'].values
            weekly_low = df_1w['low'].values
            weekly_close = df_1w['close'].values
            
            # Arrays to store weekly R1 and S1 levels
            weekly_r1 = np.full_like(weekly_close, np.nan)
            weekly_s1 = np.full_like(weekly_close, np.nan)
            
            # Calculate for each weekly bar (starting from index 1 to avoid look-ahead)
            for j in range(1, len(weekly_close)):
                range_val = weekly_high[j-1] - weekly_low[j-1]
                if range_val > 0:
                    weekly_r1[j] = weekly_close[j-1] + (range_val * 1.1 / 12)
                    weekly_s1[j] = weekly_close[j-1] - (range_val * 1.1 / 12)
            
            # Align the weekly R1/S1 to daily timeframe
            weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
            weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
            
            current_close = prices['close'].iloc[i]
            current_volume = prices['volume'].iloc[i]
            
            # Volume spike: current volume > 1.5x weekly average volume
            vol_spike = (not np.isnan(vol_avg_1w_aligned[i]) and 
                         current_volume > 1.5 * vol_avg_1w_aligned[i])
            
            if position == 0:
                # Long: price breaks above weekly R1 with volume spike
                if (not np.isnan(weekly_r1_aligned[i]) and 
                    current_close > weekly_r1_aligned[i] and vol_spike):
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below weekly S1 with volume spike
                elif (not np.isnan(weekly_s1_aligned[i]) and 
                      current_close < weekly_s1_aligned[i] and vol_spike):
                    signals[i] = -0.25
                    position = -1
            
            elif position == 1:
                # Long exit: price breaks below weekly S1 or volume dries up
                if ((not np.isnan(weekly_s1_aligned[i]) and 
                     current_close < weekly_s1_aligned[i]) or
                    (not np.isnan(vol_avg_1w_aligned[i]) and 
                     current_volume < 0.5 * vol_avg_1w_aligned[i])):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            
            elif position == -1:
                # Short exit: price breaks above weekly R1 or volume dries up
                if ((not np.isnan(weekly_r1_aligned[i]) and 
                     current_close > weekly_r1_aligned[i]) or
                    (not np.isnan(vol_avg_1w_aligned[i]) and 
                     current_volume < 0.5 * vol_avg_1w_aligned[i])):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # Not enough weekly data yet
            continue
    
    return signals
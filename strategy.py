#!/usr/bin/env python3
"""
12h_WeeklyPivot_DonchianBreakout_VolumeConfirmation_v1
Hypothesis: Weekly pivot levels act as strong support/resistance. 
Long when price breaks above weekly pivot R1 with volume confirmation and price above weekly Donchian high (20-period).
Short when price breaks below weekly pivot S1 with volume confirmation and price below weekly Donchian low.
Uses 12h timeframe for entries, weekly timeframe for pivot and Donchian levels.
Targets 50-150 total trades over 4 years (12-37/year) with strict entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Weekly Pivot Points (using weekly high/low/close) ===
    df_1w = get_htf_data(prices, '1w')
    # Calculate pivot points from weekly OHLC
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    
    # Align weekly pivot levels to 12h timeframe (with 1-bar delay for completed weekly bar)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # === Weekly Donchian Channel (20-period) ===
    # Calculate Donchian from weekly high/low
    donchian_high = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # === Volume Confirmation (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above weekly R1 AND weekly Donchian high with volume
            if (close[i] > r1_aligned[i] and 
                close[i] > donchian_high_aligned[i] and 
                vol_confirmed):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below weekly S1 AND weekly Donchian low with volume
            elif (close[i] < s1_aligned[i] and 
                  close[i] < donchian_low_aligned[i] and 
                  vol_confirmed):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses back below weekly pivot
            if close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses back above weekly pivot
            if close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyPivot_DonchianBreakout_VolumeConfirmation_v1"
timeframe = "12h"
leverage = 1.0
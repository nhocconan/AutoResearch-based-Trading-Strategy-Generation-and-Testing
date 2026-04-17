#!/usr/bin/env python3
"""
12h_WeeklyPivot_DonchianBreakout_VolumeConfirmation_v1
Weekly pivot levels from 1w combined with 12h Donchian(20) breakout and volume confirmation.
Weekly pivot acts as dynamic support/resistance, Donchian breakout confirms momentum,
volume filters noise. Works in both bull/bear markets by buying breakouts above weekly pivot
(resistance turned support) and selling breakdowns below weekly pivot (support turned resistance).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1w data for weekly pivot calculation ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points: P = (H + L + C) / 3
    # Support 1: S1 = (2 * P) - H
    # Resistance 1: R1 = (2 * P) - L
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = (2 * pivot_1w) - high_1w
    s1_1w = (2 * pivot_1w) - low_1w
    
    # Align weekly pivot levels to 12h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # === 12h Donchian(20) breakout ===
    lookback = 20
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    
    for i in range(n):
        if i >= lookback - 1:
            highest_high[i] = np.max(high[i-lookback+1:i+1])
            lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # === 12h Volume confirmation (20-period average) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(n):
        if i >= 20:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    vol_confirm = volume > vol_ma_20 * 1.5  # volume spike: 1.5x average
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 30
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above weekly R1 with volume confirmation
            if (high[i] > r1_1w_aligned[i] and 
                close[i] > highest_high[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below weekly S1 with volume confirmation
            elif (low[i] < s1_1w_aligned[i] and 
                  close[i] < lowest_low[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price breaks below weekly pivot OR Donchian low
            if (close[i] < pivot_1w_aligned[i] or 
                close[i] < lowest_low[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above weekly pivot OR Donchian high
            if (close[i] > pivot_1w_aligned[i] or 
                close[i] > highest_high[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyPivot_DonchianBreakout_VolumeConfirmation_v1"
timeframe = "12h"
leverage = 1.0
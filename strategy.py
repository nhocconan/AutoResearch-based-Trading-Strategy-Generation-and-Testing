#!/usr/bin/env python3
"""
1d_WeeklyPivot_MeanReversion_Simple
Hypothesis: Mean reversion at weekly pivot levels (S3/R3) on daily timeframe with volume confirmation.
Works in both bull and bear markets by fading extreme moves at weekly support/resistance.
Uses 1d timeframe for low trade frequency (target: 15-25 trades/year) to minimize fee drag.
Weekly pivot provides strong structural levels that hold across market regimes.
"""

name = "1d_WeeklyPivot_MeanReversion_Simple"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's values for pivot calculation
    ph = np.concatenate([[high_1w[0]], high_1w[:-1]])  # previous high
    pl = np.concatenate([[low_1w[0]], low_1w[:-1]])   # previous low
    pc = np.concatenate([[close_1w[0]], close_1w[:-1]]) # previous close
    
    # Calculate weekly pivot points
    pivot = (ph + pl + pc) / 3.0
    rang = ph - pl
    
    # Key levels: S3 and R3 (strongest support/resistance)
    s3 = pl - 2.0 * (ph - pivot)  # S3 = Low - 2*(High - Pivot)
    r3 = ph + 2.0 * (pivot - pl)  # R3 = High + 2*(Pivot - Low)
    
    # Align weekly levels to daily timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    
    # Volume confirmation: current volume / 20-day average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(20, 1)  # Ensure volume MA is ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Enter long at S3 with volume confirmation
            if (low[i] <= s3_aligned[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Enter short at R3 with volume confirmation
            elif (high[i] >= r3_aligned[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Exit long at pivot or after 5 days max hold
            if (close[i] >= pivot[i] or bars_since_entry >= 5):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short at pivot or after 5 days max hold
            if (close[i] <= pivot[i] or bars_since_entry >= 5):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals
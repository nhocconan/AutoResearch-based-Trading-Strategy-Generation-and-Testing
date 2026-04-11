#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with weekly pivot breakout + volume confirmation.
# Uses weekly pivot points (S1/S2/R1/R2) from prior week to filter breakouts.
# Long when price breaks above weekly R2 with volume > 1.5x average, short when breaks below S2.
# Designed for low trade frequency (~20-30/year) to minimize fee drag while capturing strong momentum.
# Works in bull/bear markets by only taking breakouts in the direction of weekly pivot bias.

name = "6h_1w_pivot_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's data)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot calculation (standard floor trader pivots)
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r1 = 2 * pivot_1w - low_1w
    s1 = 2 * pivot_1w - high_1w
    r2 = pivot_1w + range_1w
    s2 = pivot_1w - range_1w
    
    # Align weekly pivot levels to 6h timeframe
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Calculate weekly average volume (for confirmation)
    volume_1w = df_1w['volume'].values
    vol_avg_5_1w = pd.Series(volume_1w).rolling(window=5, min_periods=5).mean().values
    vol_avg_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_5_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 5 to ensure weekly averages are valid
    for i in range(5, n):
        # Skip if any required data is invalid
        if (np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * weekly average volume
        vol_filter = volume[i] > 1.5 * vol_avg_aligned[i]
        
        # Entry conditions: price breaks through weekly S2/R2 with volume confirmation
        long_entry = (high[i] > r2_aligned[i] and vol_filter)
        short_entry = (low[i] < s2_aligned[i] and vol_filter)
        
        # Exit conditions: price returns to weekly pivot level
        pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
        exit_long = low[i] < pivot_aligned[i] if not np.isnan(pivot_aligned[i]) else False
        exit_short = high[i] > pivot_aligned[i] if not np.isnan(pivot_aligned[i]) else False
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
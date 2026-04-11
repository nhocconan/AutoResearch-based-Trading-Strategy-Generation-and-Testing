#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1-week pivot bounce + volume confirmation.
# Uses weekly pivot points (S1/S2/R1/R2) from previous week to capture mean-reversion bounces.
# Long when price bounces above S1 with volume > 1.3x weekly average, short when rejects at R1.
# Designed for low trade frequency (~15-30/year) to minimize fee decay while capturing weekly swings.
# Weekly pivots provide stronger support/resistance than daily, reducing whipsaw in choppy markets.
# Works in bull/bear markets by fading extremes at key weekly support/resistance levels.

name = "4h_1w_pivot_bounce_volume_v1"
timeframe = "4h"
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
    
    # Calculate weekly pivot points (using previous week's data)
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
    
    # Align weekly pivot levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Calculate weekly average volume (for confirmation)
    volume_1w = df_1w['volume'].values
    vol_avg_4_1w = pd.Series(volume_1w).rolling(window=4, min_periods=4).mean().values
    vol_avg_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_4_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 4 to ensure volume average is valid
    for i in range(4, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.3 * weekly average volume
        vol_filter = volume[i] > 1.3 * vol_avg_aligned[i]
        
        # Entry conditions: price bounces off S1/S2 or rejects at R1/R2 with volume
        long_entry = ((low[i] <= s1_aligned[i] and close[i] > s1_aligned[i]) or 
                     (low[i] <= s2_aligned[i] and close[i] > s2_aligned[i])) and vol_filter
        short_entry = ((high[i] >= r1_aligned[i] and close[i] < r1_aligned[i]) or 
                      (high[i] >= r2_aligned[i] and close[i] < r2_aligned[i])) and vol_filter
        
        # Exit conditions: price reaches opposite pivot level
        exit_long = high[i] >= r1_aligned[i] if not np.isnan(r1_aligned[i]) else False
        exit_short = low[i] <= s1_aligned[i] if not np.isnan(s1_aligned[i]) else False
        
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
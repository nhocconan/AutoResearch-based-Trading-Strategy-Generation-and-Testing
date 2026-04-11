#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1-day pivot bounce + volume confirmation.
# Uses daily pivot points (S1/S2/R1/R2) from previous day to capture mean-reversion bounces.
# Long when price bounces above S1 with volume > 1.3x average, short when rejects at R1.
# Designed for low trade frequency (~20-40/year) to minimize fee decay while capturing intraday swings.
# Works in bull/bear markets by fading extremes at key daily support/resistance levels.

name = "4h_1d_pivot_bounce_volume_v1"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate daily pivot points (using previous day's data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot calculation (standard floor trader pivots)
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1 = 2 * pivot_1d - low_1d
    s1 = 2 * pivot_1d - high_1d
    r2 = pivot_1d + range_1d
    s2 = pivot_1d - range_1d
    
    # Align daily pivot levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Calculate daily average volume (for confirmation)
    volume_1d = df_1d['volume'].values
    vol_avg_10_1d = pd.Series(volume_1d).rolling(window=10, min_periods=10).mean().values
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_10_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 10 to ensure volume average is valid
    for i in range(10, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.3 * daily average volume
        vol_filter = volume[i] > 1.3 * vol_avg_aligned[i]
        
        # Entry conditions: price bounces off S1/S2 or rejects at R1/R2 with volume
        long_entry = ((low[i] <= s1_aligned[i] and close[i] > s1_aligned[i]) or 
                     (low[i] <= s2_aligned[i] and close[i] > s2_aligned[i])) and vol_filter
        short_entry = ((high[i] >= r1_aligned[i] and close[i] < r1_aligned[i]) or 
                      (high[i] >= r2_aligned[i] and close[i] < r2_aligned[i])) and vol_filter
        
        # Exit conditions: price reaches opposite pivot level
        r1_exit = align_htf_to_ltf(prices, df_1d, r1)
        s1_exit = align_htf_to_ltf(prices, df_1d, s1)
        exit_long = high[i] >= r1_exit[i] if not np.isnan(r1_exit[i]) else False
        exit_short = low[i] <= s1_exit[i] if not np.isnan(s1_exit[i]) else False
        
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
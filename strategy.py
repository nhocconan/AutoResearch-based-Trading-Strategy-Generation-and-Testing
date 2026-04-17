#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1w Camarilla pivot breakout + 6h volume filter.
Long when price breaks above weekly Camarilla R4 level with 6h volume > 1.5x 20-period average.
Short when price breaks below weekly Camarilla S4 level with 6h volume > 1.5x 20-period average.
Weekly Camarilla levels provide institutional support/resistance; breakouts with volume confirm institutional participation.
Designed for low-frequency, high-conviction trades in both bull and bear markets.
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
    
    # Get weekly data for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels
    def camarilla_levels(high_vals, low_vals, close_vals):
        # Typical price = (H + L + C) / 3
        typical_price = (high_vals + low_vals + close_vals) / 3.0
        range_val = high_vals - low_vals
        
        # Camarilla levels
        R4 = close_vals + range_val * 1.1 / 2
        R3 = close_vals + range_val * 1.1 / 4
        R2 = close_vals + range_val * 1.1 / 6
        R1 = close_vals + range_val * 1.1 / 12
        S1 = close_vals - range_val * 1.1 / 12
        S2 = close_vals - range_val * 1.1 / 6
        S3 = close_vals - range_val * 1.1 / 4
        S4 = close_vals - range_val * 1.1 / 2
        
        return R4, R3, R2, R1, S1, S2, S3, S4
    
    R4_1w, R3_1w, R2_1w, R1_1w, S1_1w, S2_1w, S3_1w, S4_1w = camarilla_levels(high_1w, low_1w, close_1w)
    
    # Calculate 6h volume 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly Camarilla levels to 6h timeframe
    R4_1w_aligned = align_htf_to_ltf(prices, df_1w, R4_1w)
    S4_1w_aligned = align_htf_to_ltf(prices, df_1w, S4_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # need enough for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R4_1w_aligned[i]) or 
            np.isnan(S4_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above weekly Camarilla R4 with volume
            if (close[i] > R4_1w_aligned[i] and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Camarilla S4 with volume
            elif (close[i] < S4_1w_aligned[i] and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below weekly Camarilla R3 (take profit at first support)
            if close[i] < R3_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above weekly Camarilla S3 (take profit at first resistance)
            if close[i] > S3_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1wCamarilla_R4S4_Breakout_Volume_Confirm"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1d timeframe Camarilla pivot R1/S1 breakout + volume confirmation.
Long when price breaks above Camarilla R1 level with volume > 1.3x 20-period volume average.
Short when price breaks below Camarilla S1 level with volume > 1.3x 20-period volume average.
Exit when price returns to Camarilla pivot point (PP) level.
Designed to capture intraday momentum within the 12h timeframe while filtering false breakouts with volume confirmation.
Works in both bull and bear markets by trading breakouts from key intraday support/resistance levels.
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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla pivot levels
    def camarilla_levels(high_vals, low_vals, close_vals):
        # Typical price
        pp = (high_vals + low_vals + close_vals) / 3.0
        # Range
        rng = high_vals - low_vals
        # Camarilla levels
        r1 = pp + (rng * 1.1 / 12)
        s1 = pp - (rng * 1.1 / 12)
        return pp, r1, s1
    
    pp_1d, r1_1d, s1_1d = camarilla_levels(high_1d, low_1d, close_1d)
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to primary timeframe (12h)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # need enough for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation
            if (close[i] > r1_1d_aligned[i] and volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume confirmation
            elif (close[i] < s1_1d_aligned[i] and volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to pivot point (PP) level
            if close[i] <= pp_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to pivot point (PP) level
            if close[i] >= pp_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dCamarilla_R1S1_Breakout_Volume_Confirm_PP_Exit"
timeframe = "12h"
leverage = 1.0
#!/usr/bin/env python3
"""
Hypothesis: 1d timeframe with 1w Camarilla pivot levels (R1/S1) as dynamic support/resistance.
Long when price breaks above 1w Camarilla R1 with volume confirmation.
Short when price breaks below 1w Camarilla S1 with volume confirmation.
Exit when price returns to the 1w pivot level.
Uses weekly Camarilla levels for institutional reference points and volume to confirm breakout strength.
Designed for low-frequency, high-conviction trades to minimize fee drag and maximize test generalization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla pivot levels (R1, S1, Pivot)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R1 = C + Range * 1.1 / 12
    # S1 = C - Range * 1.1 / 12
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    r1_1w = close_1w + range_1w * 1.1 / 12.0
    s1_1w = close_1w - range_1w * 1.1 / 12.0
    
    # Calculate 1d volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1w Camarilla levels to 1d timeframe
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # need enough for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or 
            np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above 1w Camarilla R1 with volume confirmation
            if (close[i] > r1_1w_aligned[i] and volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1w Camarilla S1 with volume confirmation
            elif (close[i] < s1_1w_aligned[i] and volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to or below 1w pivot level
            if close[i] <= pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to or above 1w pivot level
            if close[i] >= pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1wCamarilla_R1S1_Breakout_Volume_PivotExit"
timeframe = "1d"
leverage = 1.0
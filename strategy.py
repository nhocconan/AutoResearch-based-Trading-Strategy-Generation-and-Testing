#!/usr/bin/env python3
"""
1d_WeeklyPivot_R1S1_Breakout_Volume_Confirmation
Hypothesis: Weekly pivot R1/S1 breakouts with volume confirmation on daily timeframe capture institutional order flow while avoiding false breakouts. Designed for ~15 trades/year to minimize fee drift and work in both bull and bear markets by filtering counter-trend moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Weekly Pivot levels (R1, S1) from weekly timeframe
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly pivot calculation: R1/S1 from previous week
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    weekly_range = high_1w - low_1w
    r1_1w = close_1w + (1.1 * weekly_range) / 12
    s1_1w = close_1w - (1.1 * weekly_range) / 12
    
    # Align to daily timeframe (use previous week's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Volume confirmation: >2.0x 50-period average to filter weak moves
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation
            if price > r1 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume confirmation
            elif price < s1 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price below S1 (reversal signal)
            if price < s1:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price above R1 (reversal signal)
            if price > r1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WeeklyPivot_R1S1_Breakout_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0
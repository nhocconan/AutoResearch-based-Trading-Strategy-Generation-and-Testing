#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_Confirmation
Strategy: 12h Camarilla R1/S1 breakout with volume confirmation.
Long: Close > R1 and volume > 1.5x 10-period average
Short: Close < S1 and volume > 1.5x 10-period average
Exit: Close crosses back to H3 or L3 (Camarilla levels)
Position size: 0.25
Uses 1d high/low/close to calculate Camarilla levels for next 12h period.
Designed for 12h timeframe with ~15-25 trades/year.
"""

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
    
    # Get 1d data for Camarilla calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # R1 = Close + (High - Low) * 1.12
    # S1 = Close - (High - Low) * 1.12
    # H3 = Close + (High - Low) * 1.10/4
    # L3 = Close - (High - Low) * 1.10/4
    rng = high_1d - low_1d
    r1 = close_1d + rng * 1.12
    s1 = close_1d - rng * 1.12
    h3 = close_1d + rng * 1.10 / 4
    l3 = close_1d - rng * 1.10 / 4
    
    # Align Camarilla levels to 12h timeframe (they become available after 1d bar closes)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume confirmation (10-period MA on 12h)
    volume_ma10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 10  # Need volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or 
            np.isnan(volume_ma10[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 10-period average
        volume_filter = volume[i] > (1.5 * volume_ma10[i])
        
        if position == 0:
            # Long: Close > R1 and volume filter
            if close[i] > r1_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: Close < S1 and volume filter
            elif close[i] < s1_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close crosses back to H3
            if close[i] < h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close crosses back to L3
            if close[i] > l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0
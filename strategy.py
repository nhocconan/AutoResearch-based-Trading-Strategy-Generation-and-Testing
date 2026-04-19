#!/usr/bin/env python3
"""
6h_WeeklyPivot_Momentum_With_Volume_Filter
Hypothesis: Price momentum breaking above weekly pivot resistance (R1) or below support (S1) with volume confirmation
works in both bull and bear markets because weekly pivots act as significant support/resistance levels
and volume confirms institutional participation. Uses 6h timeframe to target 50-150 total trades over 4 years.
"""

name = "6h_WeeklyPivot_Momentum_With_Volume_Filter"
timeframe = "6h"
leverage = 1.0

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
    
    # Weekly data for pivot points (using 1w timeframe)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    wh = df_1w['high'].shift(1).values  # Previous week high
    wl = df_1w['low'].shift(1).values   # Previous week low
    wc = df_1w['close'].shift(1).values # Previous week close
    
    # Standard pivot point calculations
    wp = (wh + wl + wc) / 3.0           # Pivot point
    wr1 = (2 * wp) - wl                 # Resistance 1
    ws1 = (2 * wp) - wh                 # Support 1
    wr2 = wp + (wh - wl)                # Resistance 2
    ws2 = wp - (wh - wl)                # Support 2
    
    # Align weekly pivot levels to 6h timeframe
    wp_aligned = align_htf_to_ltf(prices, df_1w, wp)
    wr1_aligned = align_htf_to_ltf(prices, df_1w, wr1)
    ws1_aligned = align_htf_to_ltf(prices, df_1w, ws1)
    wr2_aligned = align_htf_to_ltf(prices, df_1w, wr2)
    ws2_aligned = align_htf_to_ltf(prices, df_1w, ws2)
    
    # Volume confirmation: volume > 1.5 * 20-period average (moderate filter)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(wp_aligned[i]) or np.isnan(wr1_aligned[i]) or 
            np.isnan(ws1_aligned[i]) or np.isnan(wr2_aligned[i]) or 
            np.isnan(ws2_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above WR1 with volume confirmation
            if (close[i] > wr1_aligned[i]) and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below WS1 with volume confirmation
            elif (close[i] < ws1_aligned[i]) and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below WP or reaches WR2 (take profit)
            if (close[i] < wp_aligned[i]) or (close[i] > wr2_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above WP or reaches WS2 (take profit)
            if (close[i] > wp_aligned[i]) or (close[i] < ws2_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
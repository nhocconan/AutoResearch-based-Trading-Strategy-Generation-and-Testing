#!/usr/bin/env python3
"""
1d_1w_Camarilla_Pivot_Breakout
Hypothesis: Uses weekly Camarilla pivot levels on Bitcoin and Ethereum with breakout confirmation. 
In both bull and bear markets, price tends to respect weekly pivot levels (H3/L3, H4/L4) as support/resistance.
Breakouts beyond these levels with volume confirmation indicate strong directional moves.
Uses daily timeframe for execution with weekly pivot levels as reference.
Target: 10-25 trades/year (40-100 total over 4 years) to minimize fee drag.
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
    
    # Get weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels
    # Formula: Based on previous week's high, low, close
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot levels for each week
    pivot = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Camarilla levels
    H4 = close_1w + (range_1w * 1.1 / 2)
    L4 = close_1w - (range_1w * 1.1 / 2)
    H3 = close_1w + (range_1w * 1.1 / 4)
    L3 = close_1w - (range_1w * 1.1 / 4)
    H2 = close_1w + (range_1w * 1.1 / 6)
    L2 = close_1w - (range_1w * 1.1 / 6)
    H1 = close_1w + (range_1w * 1.1 / 12)
    L1 = close_1w - (range_1w * 1.1 / 12)
    
    # Align weekly levels to daily timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1w, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1w, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1w, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1w, L3)
    
    # Volume confirmation: 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if any data is not ready
        if (np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or 
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume expansion: current volume > 1.5x 20-day average
        volume_expansion = volume[i] > (vol_ma_20[i] * 1.5)
        
        # Breakout conditions
        breakout_long = (close[i] > H4_aligned[i]) and volume_expansion
        breakout_short = (close[i] < L4_aligned[i]) and volume_expansion
        
        # Exit conditions: return to H3/L3 levels
        exit_long = (position == 1) and (close[i] <= H3_aligned[i])
        exit_short = (position == -1) and (close[i] >= L3_aligned[i])
        
        # Update position and signals
        if breakout_long and position != 1:
            position = 1
            signals[i] = position_size
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_1w_Camarilla_Pivot_Breakout"
timeframe = "1d"
leverage = 1.0
#!/usr/bin/env python3
"""
1d_1w_Weekly_Range_Breakout_v1
Hypothesis: Price breaks above/below weekly high/low with volume confirmation.
Weekly range breakouts capture momentum in both bull and bear markets.
Volume filter ensures breakouts are genuine. Target 30-70 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Weekly_Range_Breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for range calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly high/low (previous completed week)
    weekly_high = df_1w['high'].shift(1).values
    weekly_low = df_1w['low'].shift(1).values
    
    # Align weekly high/low to daily
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Daily volume average (20-period)
    volume_ma = np.full(n, np.nan)
    for i in range(20, n):
        volume_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirm = volume[i] > 1.5 * volume_ma[i]
        
        # Breakout conditions
        long_breakout = high[i] > weekly_high_aligned[i] and volume_confirm
        short_breakout = low[i] < weekly_low_aligned[i] and volume_confirm
        
        # Exit conditions: return to opposite weekly boundary
        long_exit = low[i] < weekly_low_aligned[i]
        short_exit = high[i] > weekly_high_aligned[i]
        
        # Signal logic
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
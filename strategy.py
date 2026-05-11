#!/usr/bin/env python3
"""
12h_1w_Pivot_MeanReversion_With_Volume
Hypothesis: Weekly pivot levels act as strong support/resistance on 12h chart. 
Price tends to revert from weekly R1/S1 levels back toward weekly pivot point.
Only take mean-reversion trades when price touches weekly R1/S1 with volume confirmation.
Works in both bull and bear markets as pivot levels are recalculated weekly and 
mean reversion occurs in ranging and trending markets. 
Target: Low frequency (12-30 trades/year) with high win rate via confluence.
"""

name = "12h_1w_Pivot_MeanReversion_With_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w data for weekly pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Calculate weekly pivot points (using prior week's OHLC) ---
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week values (shifted by 1)
    phigh = np.roll(high_1w, 1)
    plow = np.roll(low_1w, 1)
    pclose = np.roll(close_1w, 1)
    # First week has no previous, set to same week values
    phigh[0] = high_1w[0]
    plow[0] = low_1w[0]
    pclose[0] = close_1w[0]
    
    # Weekly pivot calculations
    pivot = (phigh + plow + pclose) / 3.0
    r1 = 2 * pivot - plow
    s1 = 2 * pivot - phigh
    
    # Align weekly levels to 12h
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # --- Volume Spike (12h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.8 * vol_ma.values)  # Volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Entry conditions: Touch weekly R1/S1 with volume and mean reversion expectation
        # Long when price touches S1 and reverses up (but we enter on touch with volume)
        # Short when price touches R1 and reverses down
        long_touch = (low[i] <= s1_aligned[i]) and vol_spike[i]
        short_touch = (high[i] >= r1_aligned[i]) and vol_spike[i]
        
        if position == 0:
            if long_touch:
                signals[i] = 0.25
                position = 1
            elif short_touch:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        else:
            # Exit conditions: Return to weekly pivot or opposite touch
            if position == 1:
                # Exit long when price returns to pivot or touches R1
                if (close[i] >= pivot_aligned[i]) or (high[i] >= r1_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short when price returns to pivot or touches S1
                if (close[i] <= pivot_aligned[i]) or (low[i] <= s1_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
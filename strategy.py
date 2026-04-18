#!/usr/bin/env python3
"""
4h_1w_HighLow_Reversal
Hypothesis: Uses 1-week high and low as key support/resistance levels. Enters long when price bounces off 1-week low with volume confirmation, and short when price rejects from 1-week high. Weekly levels provide strong institutional support/resistance that works in both bull and bear markets. Target: 20-35 trades/year.
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
    
    # Get 1w data for weekly high/low levels
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly high and low (using full history up to each point)
    weekly_high = np.full(len(df_1w), np.nan)
    weekly_low = np.full(len(df_1w), np.nan)
    
    for i in range(len(df_1w)):
        weekly_high[i] = np.max(df_1w['high'].values[:i+1])
        weekly_low[i] = np.min(df_1w['low'].values[:i+1])
    
    # Align weekly levels to 4h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bounce off weekly low with volume confirmation
            if (low[i] <= weekly_low_aligned[i] * 1.001 and  # allow small penetration
                close[i] > weekly_low_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: rejection from weekly high with volume confirmation
            elif (high[i] >= weekly_high_aligned[i] * 0.999 and  # allow small penetration
                  close[i] < weekly_high_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to weekly low or breaks weekly high
            if (close[i] <= weekly_low_aligned[i] or 
                close[i] >= weekly_high_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to weekly high or breaks weekly low
            if (close[i] >= weekly_high_aligned[i] or 
                close[i] <= weekly_low_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1w_HighLow_Reversal"
timeframe = "4h"
leverage = 1.0
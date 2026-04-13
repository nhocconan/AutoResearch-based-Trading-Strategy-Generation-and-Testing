#!/usr/bin/env python3
"""
1d_1w_HighLowBreakout_Volume
Hypothesis: Combines weekly high/low breakouts on 1w with volume confirmation on 1d.
In trending markets, price often breaks weekly highs/lows after consolidation. 
Volume surge confirms breakout strength. Works in both bull (break above weekly high) 
and bear (break below weekly low) markets. Target: 10-25 trades/year on 1d.
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
    
    # Get weekly data for high/low breakout levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly rolling high and low (50-period for context, but use expanding for true weekly high/low)
    # Use expanding window to capture true weekly high/low from start of data
    weekly_high = pd.Series(df_1w['high'].values).expanding().max().values
    weekly_low = pd.Series(df_1w['low'].values).expanding().min().values
    
    # Align weekly levels to daily timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Volume confirmation: 20-period average volume on 1d
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 2.0)  # Require 2x average volume
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Breakout conditions: price breaks weekly high/low with volume expansion
        breakout_up = (close[i] > weekly_high_aligned[i]) and volume_expansion[i]
        breakout_down = (close[i] < weekly_low_aligned[i]) and volume_expansion[i]
        
        # Exit conditions: reverse breakout or loss of momentum
        exit_long = (close[i] < weekly_low_aligned[i])  # Break below weekly low
        exit_short = (close[i] > weekly_high_aligned[i])  # Break above weekly high
        
        if breakout_up and position != 1:
            position = 1
            signals[i] = position_size
        elif breakout_down and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_1w_HighLowBreakout_Volume"
timeframe = "1d"
leverage = 1.0
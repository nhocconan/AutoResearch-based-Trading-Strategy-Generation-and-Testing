#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R for overbought/oversold and 6h Donchian breakout for entry.
# Williams %R identifies extremes (below -80 = oversold, above -20 = overbought) while Donchian breakout confirms momentum.
# Works in bull (buy oversold breakouts in uptrend) and bear (sell overbought breakouts in downtrend) regimes.
# Target: 50-150 trades over 4 years (12-37/year). Size: 0.25.

name = "6h_WilliamsR1d_DonchianBreakout_Extreme_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R(14)
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    # Handle division by zero (when high == low)
    williams_r_1d[highest_high_14 == lowest_low_14] = -50
    
    # Align Williams %R to 6h
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # 6h Donchian(20) for breakout levels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_1d_aligned[i]) or 
            np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R conditions
        oversold = williams_r_1d_aligned[i] < -80
        overbought = williams_r_1d_aligned[i] > -20
        
        # Breakout conditions
        long_breakout = close[i] > highest_high_20[i]
        short_breakout = close[i] < lowest_low_20[i]
        
        long_entry = oversold and long_breakout
        short_entry = overbought and short_breakout
        
        # Exit: opposite Donchian breakout (10-bar for faster exit)
        highest_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
        lowest_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
        long_exit = close[i] < highest_high_10[i]
        short_exit = close[i] > lowest_low_10[i]
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals
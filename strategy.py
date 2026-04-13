#!/usr/bin/env python3
"""
1d_1w_Range_Breakout_With_Volume
Hypothesis: Trade breakouts from weekly range on daily timeframe with volume confirmation.
Weekly range (high-low) acts as strong support/resistance. Breakouts above weekly high 
signal institutional buying; breakdowns below weekly low signal distribution.
Volume expansion confirms participation. Designed to work in both bull and bear markets
by capturing institutional moves. Target: 10-20 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for range
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Align weekly high/low to daily
    weekly_high_aligned = align_htf_to_ltf(prices, df_weekly, high_weekly)
    weekly_low_aligned = align_htf_to_ltf(prices, df_weekly, low_weekly)
    
    # Volume confirmation: current volume > 2.0x 20-period average (strict to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long: breakout above weekly high with volume expansion
        long_condition = (close[i] > weekly_high_aligned[i]) and volume_expansion[i]
        
        # Short: breakdown below weekly low with volume expansion
        short_condition = (close[i] < weekly_low_aligned[i]) and volume_expansion[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1d_1w_Range_Breakout_With_Volume"
timeframe = "1d"
leverage = 1.0
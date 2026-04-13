#!/usr/bin/env python3
"""
4h_1d_Donchian_Breakout_Volume
Hypothesis: Breakout above daily Donchian(20) high or below low with volume expansion.
Daily Donchian provides strong support/resistance. Volume confirms institutional participation.
Works in both bull (breakouts up) and bear (breakdowns down) markets. Target: 20-30 trades/year.
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
    
    # Get 1d data for Donchian levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Donchian(20) - highest high and lowest low of past 20 days
    high_max = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe
    donchian_high = align_htf_to_ltf(prices, df_1d, high_max)
    donchian_low = align_htf_to_ltf(prices, df_1d, low_min)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long: breakout above Donchian high with volume expansion
        long_condition = (close[i] > donchian_high[i]) and volume_expansion[i]
        
        # Short: breakdown below Donchian low with volume expansion
        short_condition = (close[i] < donchian_low[i]) and volume_expansion[i]
        
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

name = "4h_1d_Donchian_Breakout_Volume"
timeframe = "4h"
leverage = 1.0
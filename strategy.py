#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R extremes with 6h Donchian(20) breakout confirmation and volume filter.
# Enter long when price breaks above 6h Donchian(20) upper band, 1d Williams %R < -80 (oversold), and volume > 1.5x 20-bar average.
# Enter short when price breaks below 6h Donchian(20) lower band, 1d Williams %R > -20 (overbought), and volume > 1.5x 20-bar average.
# Exit on opposite Donchian band (lower for long exit, upper for short exit).
# Uses discrete position sizing (0.25) to control risk. Target: 50-150 total trades over 4 years.
# Williams %R identifies overextended conditions, Donchian breakout confirms directional momentum,
# volume filter ensures breakout validity. Works in bull (breakouts from oversold) and bear (failed rallies from overbought) markets.

name = "6h_WilliamsR_Donchian20_Breakout_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Donchian calculation (primary timeframe - compute directly)
    if n < 40:  # Need at least 20*2 bars for Donchian
        return np.zeros(n)
    
    # Calculate 6h Donchian(20) bands
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for Williams %R (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Williams %R(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align 1d Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 6h volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(williams_r_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R conditions: oversold (< -80) for long, overbought (> -20) for short
        williams_oversold = williams_r_aligned[i] < -80
        williams_overbought = williams_r_aligned[i] > -20
        
        # Donchian breakout conditions with volume confirmation and Williams %R filter
        long_breakout = close[i] > donchian_upper[i] and williams_oversold and volume_confirm[i]
        short_breakout = close[i] < donchian_lower[i] and williams_overbought and volume_confirm[i]
        
        # Exit conditions: opposite Donchian band
        long_exit = close[i] < donchian_lower[i]
        short_exit = close[i] > donchian_upper[i]
        
        # Handle entries and exits
        if long_breakout and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and position >= 0:
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
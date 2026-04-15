#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with weekly trend filter and volume confirmation
# Uses weekly Donchian channels to establish trend direction and daily breakouts for entry.
# Works in bull markets (breakouts above weekly high) and bear markets (breakouts below weekly low).
# Volume confirmation reduces false breakouts. Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter (Donchian channels)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Load daily data for entry signals (breakouts)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate weekly Donchian channels (20-period)
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian channels to 12h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    
    # Calculate daily Donchian channels (20-period) for entry
    high_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian channels to 12h timeframe
    high_20_1d_aligned = align_htf_to_ltf(prices, df_1d, high_20_1d)
    low_20_1d_aligned = align_htf_to_ltf(prices, df_1d, low_20_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or
            np.isnan(high_20_1d_aligned[i]) or np.isnan(low_20_1d_aligned[i])):
            continue
        
        # Long entry: price breaks above weekly high AND daily high + volume confirmation
        if (close[i] > high_20_aligned[i] and
            close[i] > high_20_1d_aligned[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-10):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below weekly low AND daily low + volume confirmation
        elif (close[i] < low_20_aligned[i] and
              close[i] < low_20_1d_aligned[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-10):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout (opposite weekly level)
        elif position == 1 and close[i] < low_20_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > high_20_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_1w_Donchian_Trend_1d_Breakout_Volume"
timeframe = "12h"
leverage = 1.0
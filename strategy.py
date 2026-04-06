#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly breakout with volume confirmation and ATR stop
# Enter long when: price breaks above weekly Donchian(10) high + volume > 1.5x average
# Enter short when: price breaks below weekly Donchian(10) low + volume > 1.5x average
# Exit when: price crosses back below/above Donchian midpoint OR opposite breakout
# Uses weekly structure to capture multi-day trends with volume confirmation
# Target: 30-80 trades over 4 years to minimize fee drag

name = "1d_weekly_breakout_vol_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly Donchian channels (10-period)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Donchian channels
    high_roll = pd.Series(high_1w).rolling(window=10, min_periods=10).max().values
    low_roll = pd.Series(low_1w).rolling(window=10, min_periods=10).min().values
    mid_roll = (high_roll + low_roll) / 2.0
    
    # Align to daily timeframe
    high_1w_aligned = align_htf_to_ltf(prices, df_1w, high_roll)
    low_1w_aligned = align_htf_to_ltf(prices, df_1w, low_roll)
    mid_1w_aligned = align_htf_to_ltf(prices, df_1w, mid_roll)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(10, n):  # Wait for Donchian to stabilize
        # Skip if required data not available
        if (np.isnan(high_1w_aligned[i]) or np.isnan(low_1w_aligned[i]) or 
            np.isnan(mid_1w_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price crosses below weekly midpoint OR opposite breakout
            if close[i] < mid_1w_aligned[i] or low[i] < low_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above weekly midpoint OR opposite breakout
            if close[i] > mid_1w_aligned[i] or high[i] > high_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries with volume confirmation
            if volume[i] > volume_threshold[i]:
                if high[i] > high_1w_aligned[i]:
                    # Bullish breakout above weekly high
                    signals[i] = 0.25
                    position = 1
                elif low[i] < low_1w_aligned[i]:
                    # Bearish breakout below weekly low
                    signals[i] = -0.25
                    position = -1
    
    return signals
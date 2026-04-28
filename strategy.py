#!/usr/bin/env python3
# Hypothesis: 6h Donchian breakout (20) with weekly pivot direction filter and volume confirmation.
# Uses weekly pivot points from 1w timeframe to establish directional bias, then looks for breakouts
# on 6h timeframe with volume confirmation. This avoids whipsaws in ranging markets by only trading
# in the direction of the weekly pivot bias. Designed for 6h to target 50-150 total trades over 4 years.
# Works in both bull and bear markets by using pivot-based directional filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_pivot_points(high, low, close):
    """Calculate classic pivot points: P = (H+L+C)/3, R1 = 2*P-L, S1 = 2*P-H, etc."""
    p = (high + low + close) / 3.0
    r1 = 2 * p - low
    s1 = 2 * p - high
    r2 = p + (high - low)
    s2 = p - (high - low)
    r3 = high + 2 * (p - low)
    s3 = low - 2 * (high - p)
    return p, r1, s1, r2, s2, r3, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot point calculation (directional bias)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:  # Need enough for pivot calculation
        return np.zeros(n)
    
    # Calculate weekly pivot points
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    p, r1, s1, r2, s2, r3, s3 = calculate_pivot_points(high_1w, low_1w, close_1w)
    
    # Use weekly bias: price above pivot = bullish bias, below = bearish bias
    weekly_bias = np.where(close_1w > p, 1, -1)  # 1 for bullish, -1 for bearish
    
    # Align weekly bias to 6h timeframe
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias)
    
    # Donchian channel (20-period) on 6h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 10)  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_bias_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Weekly bias filter
        bullish_bias = weekly_bias_aligned[i] == 1
        bearish_bias = weekly_bias_aligned[i] == -1
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous period's high
        breakout_down = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        # Entry conditions with volume confirmation and bias filter
        long_entry = bullish_bias and breakout_up and volume_filter[i]
        short_entry = bearish_bias and breakout_down and volume_filter[i]
        
        # Exit conditions: opposite breakout or loss of bias
        long_exit = (not bullish_bias) or breakout_down
        short_exit = (not bearish_bias) or breakout_up
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
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

name = "6h_Donchian_WeeklyPivotBias_VolumeFilter"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with daily pivot confirmation and volume filter
# Long when price breaks above 12-period Donchian high AND price above daily S1 pivot AND volume > 1.5x 20-period average
# Short when price breaks below 12-period Donchian low AND price below daily R1 pivot AND volume > 1.5x 20-period average
# Uses daily pivot levels for institutional reference and volume filter to avoid false breakouts.
# Target: 60-150 total trades over 4 years (15-38/year) with 0.25 position size.

name = "6h_donchian12_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (12-period on 6h)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_high = high_series.rolling(window=12, min_periods=12).max()
    donchian_low = low_series.rolling(window=12, min_periods=12).min()
    
    # Volume average (20-period)
    volume_series = pd.Series(volume)
    vol_avg = volume_series.rolling(window=20, min_periods=20).mean()
    
    # Daily pivot points from 1D data
    df_1d = get_htf_data(prices, '1d')
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate pivot points: P = (H+L+C)/3, S1 = 2P - H, R1 = 2P - L
    pivot = (daily_high + daily_low + daily_close) / 3.0
    s1 = 2 * pivot - daily_high
    r1 = 2 * pivot - daily_low
    
    # Align daily pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if daily pivot data not available
        if np.isnan(pivot_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse position or donchian breakout in opposite direction
        if position == 1:  # long position
            # Exit: price breaks below 12-period Donchian low
            if close[i] <= donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above 12-period Donchian high
            if close[i] >= donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with pivot and volume filters
            # Long: price breaks above Donchian high AND above S1 pivot AND volume > 1.5x average
            if (close[i] > donchian_high[i] and 
                close[i] > s1_aligned[i] and 
                volume[i] > 1.5 * vol_avg[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND below R1 pivot AND volume > 1.5x average
            elif (close[i] < donchian_low[i] and 
                  close[i] < r1_aligned[i] and 
                  volume[i] > 1.5 * vol_avg[i]):
                signals[i] = -0.25
                position = -1
    
    return signals
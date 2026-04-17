#!/usr/bin/env python3
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
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align daily pivot levels to 12h timeframe (use previous day's levels)
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume filter: current volume > 2.0 * 30-period average (30 periods = 15 days at 12h)
    volume_ma30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    # Simple moving average filter (12h timeframe)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(30, 20)  # Need sufficient data for volume MA and SMA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_12h[i]) or np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or
            np.isnan(volume_ma30[i]) or np.isnan(sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter - more restrictive to reduce trades
        volume_filter = volume[i] > (2.0 * volume_ma30[i])
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume and above SMA
            if (close[i] > r1_12h[i] and volume_filter and close[i] > sma_20[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with volume and below SMA
            elif (close[i] < s1_12h[i] and volume_filter and close[i] < sma_20[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below pivot point
            if close[i] < pivot_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above pivot point
            if close[i] > pivot_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_DailyPivot_Breakout_Volume_SMA"
timeframe = "12h"
leverage = 1.0
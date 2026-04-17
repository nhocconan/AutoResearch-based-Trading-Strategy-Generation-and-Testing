#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
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
    r2_1d = pivot_1d + (high_1d - low_1d)
    s2_1d = pivot_1d - (high_1d - low_1d)
    
    # Align daily pivot levels to 12h timeframe (use previous day's levels)
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average (20 periods = 10 days at 12h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Trend filter: 10-period SMA of close > 30-period SMA
    sma10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    sma30 = pd.Series(close).rolling(window=30, min_periods=30).mean().values
    trend_up = sma10 > sma30
    trend_down = sma10 < sma30
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 30  # Need sufficient data for SMA30
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_12h[i]) or np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or
            np.isnan(r2_12h[i]) or np.isnan(s2_12h[i]) or np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        if position == 0:
            # Long: price breaks above R2 with volume and trend up
            if (close[i] > r2_12h[i] and volume_filter and trend_up[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2 with volume and trend down
            elif (close[i] < s2_12h[i] and volume_filter and trend_down[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below S1 or trend reverses
            if (close[i] < s1_12h[i] or not trend_up[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above R1 or trend reverses
            if (close[i] > r1_12h[i] or not trend_down[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_DailyPivot_R2S2_Trend"
timeframe = "12h"
leverage = 1.0
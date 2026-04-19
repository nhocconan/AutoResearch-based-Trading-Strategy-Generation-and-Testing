#!/usr/bin/env python3
"""
12h_Donchian_20_Breakout_WeeklyTrend_Filter
Hypothesis: Donchian channel breakouts on 12h with weekly trend filter (price > weekly SMA50 for longs, < for shorts) 
and volume confirmation capture medium-term trends while avoiding counter-trend whipsaws. 
Weekly trend filter ensures alignment with higher-timeframe momentum, reducing false breakouts in chop.
Designed for 12h timeframe targeting 50-150 total trades over 4 years (12-37/year).
Works in bull/bear via trend-filtered breakouts and volume confirmation.
"""

name = "12h_Donchian_20_Breakout_WeeklyTrend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for weekly trend filter (SMA50 on daily)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly trend filter: SMA50 on daily data
    sma50_1d = pd.Series(df_1d['close']).rolling(window=50, min_periods=50).mean().values
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    # 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Donchian channels (20-period)
    high_20 = pd.Series(df_12h['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_12h['low']).rolling(window=20, min_periods=20).min().values
    high_20_aligned = align_htf_to_ltf(prices, df_12h, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_12h, low_20)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(sma50_1d_aligned[i]) or np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high, above weekly SMA50, with volume
            if (close[i] > high_20_aligned[i] and 
                close[i] > sma50_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, below weekly SMA50, with volume
            elif (close[i] < low_20_aligned[i] and 
                  close[i] < sma50_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below Donchian low or closes below weekly SMA50
            if (close[i] < low_20_aligned[i]) or (close[i] < sma50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above Donchian high or closes above weekly SMA50
            if (close[i] > high_20_aligned[i]) or (close[i] > sma50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
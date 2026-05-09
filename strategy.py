# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
6h Donchian(20) breakout + weekly pivot direction + volume confirmation
- Long when price breaks above Donchian(20) high AND weekly pivot > weekly MA(20) AND volume spike
- Short when price breaks below Donchian(20) low AND weekly pivot < weekly MA(20) AND volume spike
- Exit when price returns to Donchian midpoint OR weekly pivot trend reverses
- Uses daily for Donchian breakout reference, weekly for pivot/MA filter
- Designed for 6h timeframe with ~15-35 trades/year to minimize fee drag
"""

name = "6h_Donchian_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channels (20-day high/low)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels from daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 20-period rolling high/low for Donchian
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align daily Donchian levels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
    # Get weekly data for pivot and MA filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot point (typical price) and MA(20)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Weekly pivot = (H+L+C)/3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    
    # Weekly MA(20) of pivot
    weekly_pivot_ma = pd.Series(weekly_pivot).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly data to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_pivot_ma_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_ma)
    
    # Volume spike: current volume > 2.0x 30-period average volume
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean()
    vol_spike = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(weekly_pivot_ma_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Donchian breakout up + weekly pivot > MA + volume spike
            if (close[i] > donchian_high_aligned[i] and 
                weekly_pivot_aligned[i] > weekly_pivot_ma_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Donchian breakout down + weekly pivot < MA + volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  weekly_pivot_aligned[i] < weekly_pivot_ma_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR weekly pivot < MA
            if (close[i] < donchian_mid_aligned[i]) or (weekly_pivot_aligned[i] < weekly_pivot_ma_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR weekly pivot > MA
            if (close[i] > donchian_mid_aligned[i]) or (weekly_pivot_aligned[i] > weekly_pivot_ma_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
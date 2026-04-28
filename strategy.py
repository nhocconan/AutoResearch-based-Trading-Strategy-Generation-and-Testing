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
    
    # Get 12h data for Donchian and weekly pivot
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # Get 1d data for weekly pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 7:
        return np.zeros(n)
    
    # 12h Donchian(20) breakout levels
    donch_high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low_20)
    
    # Weekly pivot from 1d (using last 7 days)
    weekly_high = pd.Series(high_1d).rolling(window=7, min_periods=7).max().values
    weekly_low = pd.Series(low_1d).rolling(window=7, min_periods=7).min().values
    weekly_close = pd.Series(close_1d).rolling(window=7, min_periods=7).last().values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_range = weekly_high - weekly_low
    weekly_R1 = weekly_pivot + (weekly_range * 1.0)
    weekly_S1 = weekly_pivot - (weekly_range * 1.0)
    
    # Align weekly pivot levels to 6h
    weekly_R1_aligned = align_htf_to_ltf(prices, df_1d, weekly_R1)
    weekly_S1_aligned = align_htf_to_ltf(prices, df_1d, weekly_S1)
    
    # Volume confirmation: current 6h volume > 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20, 7)  # Donchian(20), vol MA(20), weekly(7)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or
            np.isnan(weekly_R1_aligned[i]) or
            np.isnan(weekly_S1_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        long_breakout = close[i] > donch_high_aligned[i]
        short_breakout = close[i] < donch_low_aligned[i]
        
        # Weekly pivot direction filter
        above_weekly_pivot = close[i] > weekly_R1_aligned[i]  # bullish bias
        below_weekly_pivot = close[i] < weekly_S1_aligned[i]  # bearish bias
        
        # Volume filter
        volume_filter = volume[i] > vol_ma_20[i]
        
        # Entry conditions
        long_entry = long_breakout and above_weekly_pivot and volume_filter
        short_entry = short_breakout and below_weekly_pivot and volume_filter
        
        # Exit conditions: opposite Donchian breakout
        long_exit = close[i] < donch_low_aligned[i]
        short_exit = close[i] > donch_high_aligned[i]
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
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

name = "6h_Donchian20_WeeklyPivot_VolumeFilter"
timeframe = "6h"
leverage = 1.0
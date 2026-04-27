#!/usr/bin/env python3
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
    
    # Get daily data for weekly range and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Weekly high/low (21 days for ~1 month, but using weekly approximation)
    # We'll use 5-day lookback for weekly context
    week_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(1).values
    week_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(1).values
    
    # Align weekly levels to 12h timeframe
    week_high_aligned = align_htf_to_ltf(prices, df_1d, week_high)
    week_low_aligned = align_htf_to_ltf(prices, df_1d, week_low)
    
    # Volume filter: today's volume > 1.5x 5-day average
    vol_ma_5d = pd.Series(volume_1d).rolling(window=5, min_periods=5).mean().values
    volume_filter_1d = volume_1d > (vol_ma_5d * 1.5)
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    # Price position within weekly range
    range_size = week_high_aligned - week_low_aligned
    # Avoid division by zero
    range_size = np.where(range_size == 0, 1e-10, range_size)
    position_in_range = (close - week_low_aligned) / range_size
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(week_high_aligned[i]) or np.isnan(week_low_aligned[i]) or 
            np.isnan(volume_filter_aligned[i]) or np.isnan(position_in_range[i])):
            signals[i] = 0.0
            continue
        
        # Long: near weekly low (< 20%), volume filter, and reversal from extreme
        if (position_in_range[i] < 0.2 and 
            volume_filter_aligned[i] and 
            close[i] > close[i-1]):  # slight upward momentum
            signals[i] = 0.25
            position = 1
        # Short: near weekly high (> 80%), volume filter, and reversal from extreme
        elif (position_in_range[i] > 0.8 and 
              volume_filter_aligned[i] and 
              close[i] < close[i-1]):  # slight downward momentum
            signals[i] = -0.25
            position = -1
        # Exit: return to middle of range (40-60%)
        elif position == 1 and position_in_range[i] > 0.6:
            signals[i] = 0.0
            position = 0
        elif position == -1 and position_in_range[i] < 0.4:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_WeeklyRange_VolumeFilter_Reversal"
timeframe = "12h"
leverage = 1.0
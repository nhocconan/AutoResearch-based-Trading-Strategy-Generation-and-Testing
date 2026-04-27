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
    
    # Get 1d data for weekly context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly high/low from daily data (simplified weekly)
    # Weekly high = max of last 5 daily highs, weekly low = min of last 5 daily lows
    weekly_high = np.full(len(high_1d), np.nan)
    weekly_low = np.full(len(low_1d), np.nan)
    
    for i in range(4, len(high_1d)):
        weekly_high[i] = np.max(high_1d[i-4:i+1])
        weekly_low[i] = np.min(low_1d[i-4:i+1])
    
    # Align weekly levels to 1d timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1d, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1d, weekly_low)
    
    # Price position within weekly range
    weekly_range = weekly_high_aligned - weekly_low_aligned
    weekly_range_safe = np.where(weekly_range == 0, 1, weekly_range)
    price_position = (close - weekly_low_aligned) / weekly_range_safe
    
    # Volume filter: volume > 1.5x 20-day average
    vol_ma_20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly data (5 days), volume MA (20 periods)
    start_idx = max(24, 19)  # 4 days for weekly + 19 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(price_position[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        pp = price_position[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: price in lower 30% of weekly range + volume spike (mean reversion)
            if pp < 0.3 and vol_filter:
                signals[i] = size
                position = 1
            # Short: price in upper 70% of weekly range + volume spike (fade strength)
            elif pp > 0.7 and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle 50% of range or loses momentum
            if pp >= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to middle 50% of range
            if pp <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Weekly_Range_MeanReversion_VolumeFilter"
timeframe = "1d"
leverage = 1.0
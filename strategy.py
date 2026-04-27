#!/usr/bin/env python3
"""
6h Donchian(20) breakout + weekly pivot direction + volume confirmation.
Long when: 1) Price breaks above Donchian upper band (20-period high), 2) Price > weekly pivot (bullish bias), 3) Volume > 2x 20-period average.
Short when: 1) Price breaks below Donchian lower band (20-period low), 2) Price < weekly pivot (bearish bias), 3) Volume > 2x 20-period average.
Exit when price returns to Donchian midpoint (mean reversion) or trend reverses.
Designed for 6h timeframe: targets 50-150 total trades over 4 years (12-37/year).
Works in bull markets via breakout continuation and in bear via mean reversion to midpoint.
"""

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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly high, low, close for pivot point
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot point
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate Donchian channels (20-period)
    high_max_20 = np.full(n, np.nan, dtype=np.float64)
    low_min_20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        high_max_20[i] = np.max(high[i-19:i+1])
        low_min_20[i] = np.min(low[i-19:i+1])
    
    # Donchian midpoint for exit
    donchian_mid = (high_max_20 + low_min_20) / 2.0
    
    # Volume filter: volume > 2x 20-period average
    vol_ma_20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20), weekly pivot, volume MA (20)
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        upper_band = high_max_20[i]
        lower_band = low_min_20[i]
        midpoint = donchian_mid[i]
        weekly_pivot_level = weekly_pivot_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: volume > 2x average
        vol_filter = vol_now > 2.0 * vol_avg
        
        if position == 0:
            # Long: price breaks above upper band + above weekly pivot + volume spike
            if price > upper_band and price > weekly_pivot_level and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below lower band + below weekly pivot + volume spike
            elif price < lower_band and price < weekly_pivot_level and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to midpoint (mean reversion) or breaks below lower band
            if price <= midpoint or price < lower_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to midpoint (mean reversion) or breaks above upper band
            if price >= midpoint or price > upper_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0
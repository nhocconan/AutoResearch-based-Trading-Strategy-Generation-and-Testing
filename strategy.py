#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
Long when price breaks above Donchian upper band AND weekly pivot is bullish (price > weekly pivot) AND volume > 1.5x average.
Short when price breaks below Donchian lower band AND weekly pivot is bearish (price < weekly pivot) AND volume > 1.5x average.
Exit when price reverts to Donchian midpoint (mean reversion) OR weekly pivot flips direction.
Uses 6h for price/volume/Donchian, 1w for weekly pivot to reduce whipsaw and capture major trend.
Target: 50-150 total trades over 4 years (12-37/year). Weekly pivot provides strong weekly support/resistance,
volume confirmation filters fakeouts, and Donchian breakout captures sustained moves.
Works in bull markets (captures uptrends) and bear markets (captures downtrends) by aligning with weekly bias.
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
    
    # Get 1w data for weekly pivot
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot point (standard: (H+L+C)/3)
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    
    # Calculate Donchian channels on 6h (20-period)
    # Upper band = highest high over past 20 periods
    # Lower band = lowest low over past 20 periods
    # Middle band = (upper + lower) / 2
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Volume average (20-period) on 6h
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Align 1w weekly pivot and Donchian levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    donchian_upper_aligned = align_htf_to_ltf(prices, prices, donchian_upper)  # same timeframe
    donchian_lower_aligned = align_htf_to_ltf(prices, prices, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, prices, donchian_middle)
    volume_ma_aligned = align_htf_to_ltf(prices, prices, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(donchian_middle_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        pivot = weekly_pivot_aligned[i]
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        middle = donchian_middle_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > Donchian upper AND price > weekly pivot (bullish bias) AND volume > 1.5x avg
            if price > upper and price > pivot and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: price < Donchian lower AND price < weekly pivot (bearish bias) AND volume > 1.5x avg
            elif price < lower and price < pivot and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Donchian midpoint (mean reversion) OR price < weekly pivot (bias flip)
            if price < middle or price < pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Donchian midpoint (mean reversion) OR price > weekly pivot (bias flip)
            if price > middle or price > pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Volume_Filter"
timeframe = "6h"
leverage = 1.0
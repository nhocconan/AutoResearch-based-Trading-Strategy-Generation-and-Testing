#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
Long when price breaks above 6h Donchian upper band AND weekly pivot shows bullish bias (price > weekly pivot) AND volume > 1.5x average.
Short when price breaks below 6h Donchian lower band AND weekly pivot shows bearish bias (price < weekly pivot) AND volume > 1.5x average.
Exit when price reverts to 6h Donchian middle band (20-period midpoint) or weekly pivot bias reverses.
Uses 6h timeframe to reduce trade frequency vs lower timeframes, with weekly pivot providing structural bias from higher timeframe.
Target: 50-150 total trades over 4 years (12-37/year) to stay within proven working range for 6h strategies.
"""

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
    
    # Calculate 6h Donchian channels (20-period) - ONCE before loop
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_ma
    donchian_lower = low_ma
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Load weekly data for pivot point - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot point (standard: (H+L+C)/3)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align HTF indicators to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Volume average (20-period) on primary timeframe
    vol_ma_primary = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_middle[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma_primary[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        middle = donchian_middle[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        vol_ma_val = vol_ma_primary[i]
        
        # Get current price and volume
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above 6h Donchian upper AND price > weekly pivot (bullish bias) AND volume > 1.5x average
            if (price > upper and price > weekly_pivot_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 6h Donchian lower AND price < weekly pivot (bearish bias) AND volume > 1.5x average
            elif (price < lower and price < weekly_pivot_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to 6h Donchian middle OR price < weekly pivot (bias reversal)
                if price <= middle or price < weekly_pivot_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to 6h Donchian middle OR price > weekly pivot (bias reversal)
                if price >= middle or price > weekly_pivot_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
# Donchian breakout captures breakout momentum
# Weekly pivot (from 1w) determines primary trend: price > weekly pivot = bullish, < weekly pivot = bearish
# Only take breakouts in direction of weekly trend (long on bullish, short on bearish)
# Volume confirmation: volume > 1.5x 20-period average to filter false breakouts
# Works in both bull and bear markets by aligning with weekly trend direction
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load weekly data for pivot direction
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot point: (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Load 6h data for Donchian and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 20-period Donchian channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in indicators
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(pivot_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        is_bullish = close[i] > pivot_1w_aligned[i]
        is_bearish = close[i] < pivot_1w_aligned[i]
        
        # Volume confirmation
        has_volume = vol_filter[i]
        
        price = close[i]
        
        if position == 0:
            # Enter long: bullish weekly trend + price breaks above Donchian high + volume
            long_signal = False
            if is_bullish and has_volume and price > highest_high[i]:
                long_signal = True
            
            # Enter short: bearish weekly trend + price breaks below Donchian low + volume
            short_signal = False
            if is_bearish and has_volume and price < lowest_low[i]:
                short_signal = True
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below Donchian high or weekly trend turns bearish
            exit_signal = False
            if price < highest_high[i] or not is_bullish:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above Donchian low or weekly trend turns bullish
            exit_signal = False
            if price > lowest_low[i] or not is_bearish:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivotDirection_Volume"
timeframe = "6h"
leverage = 1.0
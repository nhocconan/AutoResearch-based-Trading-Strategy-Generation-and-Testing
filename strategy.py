#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Long when: price breaks above 6h Donchian upper (20-bar high), price > weekly pivot point, volume > 1.5x 20-bar avg
# Short when: price breaks below 6h Donchian lower (20-bar low), price < weekly pivot point, volume > 1.5x 20-bar avg
# Exit when price returns to 6h Donchian midpoint (mean reversion) or opposite breakout
# Weekly pivot provides structural bias (bull/bear) from higher timeframe, effective in both trending and ranging markets.
# Timeframe: 6h, HTF: 1w. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Donchian20_Breakout_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Calculate volume confirmation on 6h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1w data ONCE before loop for weekly pivot
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot point (standard formula: (H+L+C)/3)
    # Using previous week's values (shifted by 1) to avoid look-ahead
    if len(high_1w) >= 2:
        prev_high = np.roll(high_1w, 1)
        prev_low = np.roll(low_1w, 1)
        prev_close = np.roll(close_1w, 1)
        prev_high[0] = np.nan  # First value has no previous
        prev_low[0] = np.nan
        prev_close[0] = np.nan
        
        weekly_pivot = (prev_high + prev_low + prev_close) / 3.0
    else:
        weekly_pivot = np.full(len(close_1w), np.nan)
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate 6h Donchian channels (20-period)
    if len(high) >= 20:
        # Donchian upper: 20-period high
        donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        # Donchian lower: 20-period low
        donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
        # Donchian midpoint: (upper + lower) / 2
        donchian_mid = (donchian_upper + donchian_lower) / 2.0
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper, above weekly pivot, volume filter
            if (close[i] > donchian_upper[i] and 
                open_price[i] <= donchian_upper[i] and  # Ensure breakout happens on this bar
                close[i] > weekly_pivot_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower, below weekly pivot, volume filter
            elif (close[i] < donchian_lower[i] and 
                  open_price[i] >= donchian_lower[i] and  # Ensure breakdown happens on this bar
                  close[i] < weekly_pivot_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below Donchian midpoint (mean reversion) or breaks below Donchian lower (reversal)
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Donchian midpoint (mean reversion) or breaks above Donchian upper (reversal)
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
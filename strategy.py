#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
# Uses 20-period Donchian channels for breakout detection. Trade only in direction of weekly pivot
# (bullish if price above weekly pivot, bearish if below). Volume confirmation requires current volume
# > 1.5x median of past 20 periods. Works in bull markets (long breakouts above weekly pivot) and
# bear markets (short breakouts below weekly pivot). Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # Support 1 = (2 * Pivot) - High
    # Resistance 1 = (2 * Pivot) - Low
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # We only need the pivot level for direction, not full support/resistance
    
    # Align weekly pivot to 6h timeframe (no additional delay needed for pivot)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Calculate Donchian channels (20-period) on 6h
    # Upper channel = highest high of past 20 periods
    # Lower channel = lowest low of past 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is NaN
        if (np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i])):
            continue
        
        # Long entry: price breaks above Donchian upper + price above weekly pivot + volume confirmation
        if (close[i] > donchian_upper[i] and
            close[i] > pivot_1w_aligned[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian lower + price below weekly pivot + volume confirmation
        elif (close[i] < donchian_lower[i] and
              close[i] < pivot_1w_aligned[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout (opposite Donchian channel break)
        elif position == 1 and close[i] < donchian_lower[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > donchian_upper[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_Donchian_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0
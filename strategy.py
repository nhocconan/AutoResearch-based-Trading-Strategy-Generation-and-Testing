#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian breakout with weekly trend filter and volume confirmation
# Uses weekly Donchian channels for trend direction and daily breakouts for entry.
# Works in bull markets (breakouts above weekly high) and bear markets (breakouts below weekly low).
# Volume confirmation filters false breakouts. Target: 30-100 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for Donchian trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Calculate weekly Donchian channels (20-period)
    high_max = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian to daily timeframe
    high_max_aligned = align_htf_to_ltf(prices, df_weekly, high_max)
    low_min_aligned = align_htf_to_ltf(prices, df_weekly, low_min)
    
    # Calculate daily Donchian breakout levels (previous day's high/low)
    # Shift by 1 to avoid look-ahead: use previous day's levels
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max_aligned[i]) or np.isnan(low_min_aligned[i]) or
            np.isnan(prev_high[i]) or np.isnan(prev_low[i])):
            continue
        
        # Long entry: price breaks above previous day's high AND above weekly Donchian high
        # with volume confirmation
        if (close[i] > prev_high[i] and close[i] > high_max_aligned[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below previous day's low AND below weekly Donchian low
        # with volume confirmation
        elif (close[i] < prev_low[i] and close[i] < low_min_aligned[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout (opposite Donchian break)
        elif position == 1 and close[i] < low_min_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > high_max_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Long when price breaks above Donchian high(20) AND weekly pivot indicates bullish bias AND volume > 1.5x average
# Short when price breaks below Donchian low(20) AND weekly pivot indicates bearish bias AND volume > 1.5x average
# Weekly pivot bias: price above weekly pivot = bullish, below = bearish
# Works in bull markets (breakouts up with bullish bias) and bear markets (breakouts down with bearish bias)
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points (using prior week's data)
    # Weekly high/low/close from prior week
    weekly_high = np.roll(high_1d, 5)  # Approximate: 5 trading days back
    weekly_low = np.roll(low_1d, 5)
    weekly_close = np.roll(close_1d, 5)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Calculate Donchian channels (20-period) on 6h data
    # Highest high of last 20 periods
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lowest low of last 20 periods
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(weekly_pivot_aligned[i])):
            continue
        
        # Long entry: price breaks above Donchian high AND price above weekly pivot (bullish bias) AND volume confirmation
        if (close[i] > donchian_high[i] and
            close[i] > weekly_pivot_aligned[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian low AND price below weekly pivot (bearish bias) AND volume confirmation
        elif (close[i] < donchian_low[i] and
              close[i] < weekly_pivot_aligned[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse Donchian breakout or price crosses weekly pivot in opposite direction
        elif position == 1 and (close[i] < donchian_low[i] or close[i] < weekly_pivot_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > donchian_high[i] or close[i] > weekly_pivot_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_Donchian_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0
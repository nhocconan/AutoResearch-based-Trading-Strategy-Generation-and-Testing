#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
# Uses the previous 20-bar high/low as support/resistance. Breakouts are traded only when
# confirmed by volume (1.5x median) and weekly pivot bias (price above/below weekly pivot).
# Works in bull markets (breakouts up with bullish weekly bias) and bear markets
# (breakouts down with bearish weekly bias). Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot (using last 5 daily candles approx 1 week)
    # Weekly high = max of last 5 daily highs
    # Weekly low = min of last 5 daily lows
    # Weekly close = last daily close
    weekly_high = np.full(len(close_1d), np.nan)
    weekly_low = np.full(len(close_1d), np.nan)
    weekly_close = np.full(len(close_1d), np.nan)
    
    for i in range(4, len(close_1d)):
        weekly_high[i] = np.max(high_1d[i-4:i+1])
        weekly_low[i] = np.min(low_1d[i-4:i+1])
        weekly_close[i] = close_1d[i]
    
    # Weekly pivot = (weekly high + weekly low + weekly close) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Calculate Donchian channels (20-period) on 6h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(weekly_pivot_aligned[i])):
            continue
        
        # Long entry: price breaks above Donchian high + volume confirmation + price > weekly pivot
        if (close[i] > highest_high[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            close[i] > weekly_pivot_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian low + volume confirmation + price < weekly pivot
        elif (close[i] < lowest_low[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              close[i] < weekly_pivot_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse Donchian breakout
        elif position == 1 and close[i] < lowest_low[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > highest_high[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_Donchian_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0
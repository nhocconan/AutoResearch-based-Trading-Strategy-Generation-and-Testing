#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with weekly pivot direction and daily volume confirmation.
# Uses weekly pivot point (calculated from prior week) for trend bias.
# Enters long when price > weekly pivot and breaks daily Donchian(10) high with volume spike.
# Enters short when price < weekly pivot and breaks daily Donchian(10) low with volume spike.
# Exits when price crosses back across weekly pivot.
# Weekly pivot provides macro bias; daily Donchian + volume provides precise entry.
# Designed for low turnover: target 12-30 trades/year (50-120 total over 4 years).
# Works in bull/bear by following weekly pivot trend.
name = "6h_WeeklyPivot_DailyDonchian10_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Get weekly data for pivot point (prior week's OHLC)
    df_week = get_htf_data(prices, '1w')
    if len(df_week) == 0:
        return np.zeros(n)
    
    # Calculate weekly pivot point: (H + L + C) / 3 from prior week
    high_week = df_week['high'].values
    low_week = df_week['low'].values
    close_week = df_week['close'].values
    
    # Pivot point from previous week (avoid look-ahead: use shift(1))
    pivot_week = (high_week + low_week + close_week) / 3.0
    # Shift to use prior week's pivot (current week's pivot not yet known)
    pivot_week = np.roll(pivot_week, 1)
    pivot_week[0] = np.nan  # First week has no prior
    
    # Align weekly pivot to 6h timeframe (wait for weekly bar to close)
    pivot_week_aligned = align_htf_to_ltf(prices, df_week, pivot_week)
    
    # Get daily data for Donchian channels and volume
    df_day = get_htf_data(prices, '1d')
    high_day = df_day['high'].values
    low_day = df_day['low'].values
    vol_day = df_day['volume'].values
    
    # Daily Donchian(10) - use prior day's levels to avoid look-ahead
    high_10_day = pd.Series(high_day).rolling(window=10, min_periods=10).max().values
    low_10_day = pd.Series(low_day).rolling(window=10, min_periods=10).min().values
    # Shift to use prior day's levels
    high_10_day = np.roll(high_10_day, 1)
    low_10_day = np.roll(low_10_day, 1)
    high_10_day[0] = np.nan
    low_10_day[0] = np.nan
    
    # Align daily Donchian to 6h timeframe
    high_10_day_aligned = align_htf_to_ltf(prices, df_day, high_10_day)
    low_10_day_aligned = align_htf_to_ltf(prices, df_day, low_10_day)
    
    # Daily volume filter: volume > 2.0 * 20-day average
    vol_ma_20 = pd.Series(vol_day).rolling(window=20, min_periods=20).mean().values
    vol_filter = vol_day > (vol_ma_20 * 2.0)
    vol_filter_aligned = align_htf_to_ltf(prices, df_day, vol_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_week_aligned[i]) or np.isnan(high_10_day_aligned[i]) or 
            np.isnan(low_10_day_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(vol_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above weekly pivot AND breaks daily Donchian high with volume spike
            if (close[i] > pivot_week_aligned[i] and 
                close[i] > high_10_day_aligned[i] and 
                vol_filter_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly pivot AND breaks daily Donchian low with volume spike
            elif (close[i] < pivot_week_aligned[i] and 
                  close[i] < low_10_day_aligned[i] and 
                  vol_filter_aligned[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below weekly pivot
            if close[i] < pivot_week_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above weekly pivot
            if close[i] > pivot_week_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
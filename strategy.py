#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Donchian breakouts capture momentum with clear entry/exit levels.
# Weekly pivot (from prior week) provides institutional context: long only when price above weekly pivot,
# short only when below weekly pivot. This avoids counter-trend trades in strong weekly trends.
# Volume confirmation ensures breakouts have participation, reducing false signals.
# Target: 20-40 total trades over 4 years (5-10/year) to minimize fee drag on 6h timeframe.

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
    
    # Get daily data for Donchian calculation (more stable than 6h for breakout calculation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Weekly pivot from previous week: (H + L + C) / 3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    prev_close_1w[0] = np.nan
    
    weekly_pivot = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Donchian channels from daily data (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper band: 20-day high
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-day low
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume filter: volume > 1.5 x 20-period average (moderate threshold)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly pivot (1 week), Donchian (20 days), volume MA (20)
    start_idx = max(1, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: moderate volume surge
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Weekly pivot filter
        above_weekly_pivot = price > weekly_pivot_aligned[i]
        below_weekly_pivot = price < weekly_pivot_aligned[i]
        
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + volume + above weekly pivot
            if price > donchian_high_val and vol_filter and above_weekly_pivot:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low + volume + below weekly pivot
            elif price < donchian_low_val and vol_filter and below_weekly_pivot:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low or falls below weekly pivot
            if price < donchian_low_val or not above_weekly_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above Donchian high or rises above weekly pivot
            if price > donchian_high_val or not below_weekly_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0
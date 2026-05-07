#!/usr/bin/env python3
name = "6h_Donchian20_WeeklyPivot_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian and pivot calculations
    df_wk = get_htf_data(prices, '1w')
    if len(df_wk) < 50:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period high/low)
    high_max = pd.Series(df_wk['high'].values).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(df_wk['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly pivot points (standard formula)
    high_prev_wk = df_wk['high'].shift(1).values
    low_prev_wk = df_wk['low'].shift(1).values
    close_prev_wk = df_wk['close'].shift(1).values
    
    pivot = (high_prev_wk + low_prev_wk + close_prev_wk) / 3
    r1 = 2 * pivot - low_prev_wk
    s1 = 2 * pivot - high_prev_wk
    r2 = pivot + (high_prev_wk - low_prev_wk)
    s2 = pivot - (high_prev_wk - low_prev_wk)
    r3 = high_prev_wk + 2 * (pivot - low_prev_wk)
    s3 = low_prev_wk - 2 * (high_prev_wk - pivot)
    
    # Align weekly levels to 6h timeframe
    high_max_aligned = align_htf_to_ltf(prices, df_wk, high_max)
    low_min_aligned = align_htf_to_ltf(prices, df_wk, low_min)
    pivot_aligned = align_htf_to_ltf(prices, df_wk, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_wk, r1)
    s1_aligned = align_htf_to_ltf(prices, df_wk, s1)
    r2_aligned = align_htf_to_ltf(prices, df_wk, r2)
    s2_aligned = align_htf_to_ltf(prices, df_wk, s2)
    r3_aligned = align_htf_to_ltf(prices, df_wk, r3)
    s3_aligned = align_htf_to_ltf(prices, df_wk, s3)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # ~1 day for 6h to reduce trades
    
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_max_aligned[i]) or 
            np.isnan(low_min_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine trend based on price relative to weekly pivot
        trend_up = close[i] > pivot_aligned[i]
        trend_down = close[i] < pivot_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Break above weekly Donchian high in uptrend with volume
            if (close[i] > high_max_aligned[i] and 
                trend_up and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Break below weekly Donchian low in downtrend with volume
            elif (close[i] < low_min_aligned[i] and 
                  trend_down and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price re-enters weekly Donchian range or trend change
            if (close[i] < high_max_aligned[i] and close[i] > low_min_aligned[i]) or not trend_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price re-enters weekly Donchian range or trend change
            if (close[i] < high_max_aligned[i] and close[i] > low_min_aligned[i]) or not trend_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Using weekly Donchian breakouts with weekly pivot trend filter and volume confirmation
# on 6h timeframe will capture major trend changes while avoiding whipsaws. Weekly pivot acts as
# a dynamic trend filter (price above pivot = uptrend, below = downtrend). Donchian channels
# provide structural breakout levels. Volume confirmation ensures institutional participation.
# Designed for 15-35 trades per year (60-140 total over 4 years) to minimize fee drag.
# Position size of 0.25 manages drawdown, and cooldown of 4 bars prevents overtrading.
# Works in both bull and bear markets by following the weekly trend direction.
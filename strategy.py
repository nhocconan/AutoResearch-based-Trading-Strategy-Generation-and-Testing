#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivotDirection_Volume
Hypothesis: On 6h timeframe, buy when price breaks above 20-period Donchian high with weekly pivot support (price > weekly pivot) and volume confirmation; sell when breaks below 20-period Donchian low with weekly pivot resistance (price < weekly pivot) and volume confirmation. Uses weekly pivot for trend filter to avoid whipsaws and volume spike for confirmation. Weekly pivot calculated from prior week's OHLC. Designed to work in both bull and bear markets by using weekly pivot as dynamic support/resistance and Donchian breakouts for trend continuation.
"""
name = "6h_Donchian20_WeeklyPivotDirection_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week's OHLC
    # Pivot = (High + Low + Close) / 3
    weekly_high = df_weekly['high'].shift(1).values
    weekly_low = df_weekly['low'].shift(1).values
    weekly_close = df_weekly['close'].shift(1).values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume filter: current volume > 1.5 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(lookback, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data is not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        if position == 0:
            # Minimum 10 bars between trades to reduce frequency (6h timeframe)
            if bars_since_entry < 10:
                continue
                
            # Long: price breaks above Donchian high + price above weekly pivot + volume filter
            if (close[i] > donchian_high[i] and 
                close[i] > weekly_pivot_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: price breaks below Donchian low + price below weekly pivot + volume filter
            elif (close[i] < donchian_low[i] and 
                  close[i] < weekly_pivot_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position != 0:
            # Exit: price returns to opposite Donchian level
            if position == 1:
                if close[i] < donchian_low[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donchian_high[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals
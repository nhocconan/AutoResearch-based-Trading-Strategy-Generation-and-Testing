#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout + daily pivot direction + volume confirmation.
Long when price breaks above Donchian(20) high + daily close > daily pivot + volume > 1.5x average.
Short when price breaks below Donchian(20) low + daily close < daily pivot + volume > 1.5x average.
Exit when price breaks below Donchian(10) low (long) or above Donchian(10) high (short) or daily pivot direction flips.
Designed for low trade frequency (~15-25/year) to minimize fee drag in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data for pivot - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot (standard: (H+L+C)/3)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    daily_pivot = (daily_high + daily_low + daily_close) / 3.0
    daily_pivot_aligned = align_htf_to_ltf(prices, df_1d, daily_pivot)
    
    # Calculate Donchian channels (20 and 10)
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Calculate average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or
            np.isnan(donchian_high_10[i]) or np.isnan(donchian_low_10[i]) or
            np.isnan(daily_pivot_aligned[i]) or np.isnan(avg_volume[i]) or volume[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        daily_close_val = None
        daily_pivot_val = None
        if i < len(daily_pivot_aligned):
            daily_close_val = df_1d['close'].values[-1] if len(df_1d) > 0 else np.nan
            daily_pivot_val = daily_pivot_aligned[i]
        else:
            daily_close_val = np.nan
            daily_pivot_val = np.nan
            
        if np.isnan(daily_close_val) or np.isnan(daily_pivot_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        daily_trend_up = daily_close_val > daily_pivot_val
        daily_trend_down = daily_close_val < daily_pivot_val
        
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 0:
            # Long: Break above Donchian(20) high + daily uptrend + volume confirmation
            if (close[i] > donchian_high_20[i] and 
                daily_trend_up and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian(20) low + daily downtrend + volume confirmation
            elif (close[i] < donchian_low_20[i] and 
                  daily_trend_down and volume_confirm):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Break below Donchian(10) low or daily trend changes to down
                if close[i] < donchian_low_10[i] or not daily_trend_up:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Break above Donchian(10) high or daily trend changes to up
                if close[i] > donchian_high_10[i] or not daily_trend_down:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_DailyPivot_VolumeFilter"
timeframe = "6h"
leverage = 1.0
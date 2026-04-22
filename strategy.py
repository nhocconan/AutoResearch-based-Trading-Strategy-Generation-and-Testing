#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for Donchian(20) - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian(20) channels
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    upper_20 = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian channels to 6h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_weekly, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_weekly, lower_20)
    
    # Load daily data for pivot calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points (based on previous daily bar)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily_prev = df_daily['close'].values
    pp = (high_daily + low_daily + close_daily_prev) / 3.0
    pp_aligned = align_htf_to_ltf(prices, df_daily, pp)
    
    # Calculate 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly Donchian upper AND price > daily pivot with volume
            if (close[i] > upper_20_aligned[i] and 
                close[i] > pp_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly Donchian lower AND price < daily pivot with volume
            elif (close[i] < lower_20_aligned[i] and 
                  close[i] < pp_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to the opposite weekly Donchian channel or daily pivot
            if position == 1:
                if close[i] < lower_20_aligned[i] or close[i] < pp_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > upper_20_aligned[i] or close[i] > pp_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_WeeklyDonchian20_DailyPivot_Trend_Volume_Session"
timeframe = "6h"
leverage = 1.0
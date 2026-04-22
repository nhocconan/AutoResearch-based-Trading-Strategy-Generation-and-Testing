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
    
    # Calculate Donchian(20) channels from weekly data
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    upper_20 = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 12h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_weekly, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_weekly, lower_20)
    
    # Load daily data for volume average (20-period) - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate daily volume average
    daily_volume = df_daily['volume'].values
    vol_avg_20 = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20)
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i])):
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
            # Long: Price breaks above weekly Donchian(20) with volume
            if (close[i] > upper_20_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly Donchian(20) with volume
            elif (close[i] < lower_20_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to the opposite Donchian channel
            if position == 1:
                if close[i] < lower_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > upper_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12H_WeeklyDonchian20_Volume_Session"
timeframe = "12h"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy combining 1-week Donchian breakout with 1-day volume confirmation
# Uses longer-term structure (weekly) to filter direction and shorter-term volume to confirm momentum
# Designed to work in both bull and bear markets by requiring breakouts with volume in direction of weekly trend
# Target: 50-150 total trades over 4 years (12-37/year) with discrete position sizing to minimize fee drag

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
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_upper_20 = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    weekly_lower_20 = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian channels to 6h timeframe
    weekly_upper_20_aligned = align_htf_to_ltf(prices, df_weekly, weekly_upper_20)
    weekly_lower_20_aligned = align_htf_to_ltf(prices, df_weekly, weekly_lower_20)
    
    # Load daily data for volume confirmation - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate daily volume average (20-period)
    daily_volume = df_daily['volume'].values
    daily_vol_avg_20 = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily volume average to 6h timeframe
    daily_vol_avg_20_aligned = align_htf_to_ltf(prices, df_daily, daily_vol_avg_20)
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(weekly_upper_20_aligned[i]) or np.isnan(weekly_lower_20_aligned[i]) or 
            np.isnan(daily_vol_avg_20_aligned[i])):
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
            # Long: Price breaks above weekly upper Donchian(20) with above-average daily volume
            if (close[i] > weekly_upper_20_aligned[i] and 
                volume[i] > 1.5 * daily_vol_avg_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly lower Donchian(20) with above-average daily volume
            elif (close[i] < weekly_lower_20_aligned[i] and 
                  volume[i] > 1.5 * daily_vol_avg_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to the opposite weekly Donchian channel
            if position == 1:
                if close[i] < weekly_lower_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > weekly_upper_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_WeeklyDonchian20_DailyVolume"
timeframe = "6h"
leverage = 1.0
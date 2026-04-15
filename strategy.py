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
    
    # Get weekly data for trend filter
    weekly = get_htf_data(prices, '1w')
    weekly_high = weekly['high'].values
    weekly_low = weekly['low'].values
    weekly_close = weekly['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    def donchian_channels(high_arr, low_arr, window):
        upper = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    weekly_dc_upper, weekly_dc_lower = donchian_channels(weekly_high, weekly_low, 20)
    
    # Align weekly Donchian channels to 6h timeframe
    weekly_dc_upper_aligned = align_htf_to_ltf(prices, weekly, weekly_dc_upper)
    weekly_dc_lower_aligned = align_htf_to_ltf(prices, weekly, weekly_dc_lower)
    
    # Get daily data for volume confirmation
    daily = get_htf_data(prices, '1d')
    daily_volume = daily['volume'].values
    
    # Calculate daily volume average (20-period)
    daily_vol_ma = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    daily_vol_ma_aligned = align_htf_to_ltf(prices, daily, daily_vol_ma)
    
    # Volume filter: current 6h volume > 1.5x daily average volume
    volume_filter = volume > (1.5 * daily_vol_ma_aligned)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_dc_upper_aligned[i]) or np.isnan(weekly_dc_lower_aligned[i]) or 
            np.isnan(daily_vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when volume filter passes
        if volume_filter[i]:
            # Long conditions: price breaks above weekly Donchian upper with volume
            if close[i] > weekly_dc_upper_aligned[i]:
                signals[i] = 0.25
            # Short conditions: price breaks below weekly Donchian lower with volume
            elif close[i] < weekly_dc_lower_aligned[i]:
                signals[i] = -0.25
            else:
                signals[i] = signals[i-1]
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_WeeklyDonchianBreakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0
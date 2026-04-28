#!/usr/bin/env python3
# Hypothesis: 1d Donchian breakout with weekly trend filter and volume confirmation.
# Uses weekly Donchian channels (20-period) to identify trend direction and strength.
# Combines with 1d Donchian breakout (20-period) and volume spike (2x 20-period average) for entry.
# Designed for 1d timeframe with ~30-100 total trades over 4 years to minimize fee drift.
# Should work in both bull and bear markets by filtering for trend alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly upper/lower bands
    weekly_upper = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    weekly_lower = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Weekly trend: price above upper = uptrend, below lower = downtrend
    weekly_upper_aligned = align_htf_to_ltf(prices, df_1w, weekly_upper)
    weekly_lower_aligned = align_htf_to_ltf(prices, df_1w, weekly_lower)
    
    # 1d Donchian breakout (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_upper_aligned[i]) or np.isnan(weekly_lower_aligned[i]) or
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > weekly_upper_aligned[i]
        weekly_downtrend = close[i] < weekly_lower_aligned[i]
        
        # 1d Donchian breakout
        breakout_up = close[i] > high_20[i-1]  # Break above previous 20-period high
        breakout_down = close[i] < low_20[i-1]  # Break below previous 20-period low
        
        # Entry conditions
        long_entry = weekly_uptrend and breakout_up and volume_spike[i]
        short_entry = weekly_downtrend and breakout_down and volume_spike[i]
        
        # Exit conditions: opposite breakout or trend reversal
        long_exit = breakout_down or not weekly_uptrend
        short_exit = breakout_up or not weekly_downtrend
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_DonchianBreakout_1wTrendFilter_VolumeSpike"
timeframe = "1d"
leverage = 1.0
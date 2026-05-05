#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction filter and volume spike confirmation
# Long when price breaks above Donchian(20) high AND 1d price > weekly pivot point (PP) AND volume > 2.0x 20-period average
# Short when price breaks below Donchian(20) low AND 1d price < weekly pivot point (PP) AND volume > 2.0x 20-period average
# Exit when price crosses Donchian(20) midpoint OR weekly pivot direction flips
# Weekly pivot points calculated from prior 1d week (Monday-Friday): PP = (Weekly High + Weekly Low + Weekly Close) / 3
# Donchian provides clear breakout levels, weekly pivot gives higher timeframe directional bias, volume confirms participation
# Target: 12-37 trades/year per symbol (50-150 total over 4 years) for 6h timeframe
# Discrete sizing (0.25) to limit fee drag

name = "6h_Donchian20_1dWeeklyPivot_Direction_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need at least a week of data
        return np.zeros(n)
    
    # Calculate weekly pivot points from 1d data
    # Resample 1d to weekly using actual weekly boundaries (no synthetic resampling)
    # We'll calculate weekly high, low, close from the 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly aggregates manually to avoid look-ahead
    weekly_high = np.full(len(high_1d), np.nan)
    weekly_low = np.full(len(low_1d), np.nan)
    weekly_close = np.full(len(close_1d), np.nan)
    
    # Group 1d data into weeks (assuming 5 trading days per week)
    for i in range(4, len(high_1d)):  # Start from index 4 to have 5 days (0-4)
        if i % 5 == 4:  # End of week (Friday)
            week_start = i - 4
            weekly_high[i] = np.max(high_1d[week_start:i+1])
            weekly_low[i] = np.min(low_1d[week_start:i+1])
            weekly_close[i] = close_1d[i]  # Friday's close
    
    # Weekly pivot point: PP = (Weekly High + Weekly Low + Weekly Close) / 3
    weekly_pp = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly PP to 6h timeframe (wait for weekly close)
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1d, weekly_pp, additional_delay_bars=0)
    
    # Calculate Donchian(20) channels
    if len(high) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_high + donchian_low) / 2.0
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Volume confirmation: volume > 2.0x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(weekly_pp_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Donchian breakout up AND price > weekly PP AND volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > weekly_pp_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Donchian breakout down AND price < weekly PP AND volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < weekly_pp_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < Donchian midpoint OR price < weekly PP (trend flip)
            if (close[i] < donchian_mid[i] or 
                close[i] < weekly_pp_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > Donchian midpoint OR price > weekly PP (trend flip)
            if (close[i] > donchian_mid[i] or 
                close[i] > weekly_pp_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
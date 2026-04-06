#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d pivot level confirmation and volume filter
# Long when price breaks above Donchian high(20) AND above 1d weekly pivot R2 AND volume > 1.5x avg
# Short when price breaks below Donchian low(20) AND below 1d weekly pivot S2 AND volume > 1.5x avg
# Exit when price crosses opposite Donchian level or volume drops
# Uses 6h timeframe to reduce trade frequency, targets 50-150 total trades over 4 years
# Works in both bull/bear by using volatility breakouts with pivot level filtering

name = "6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate weekly pivot points from previous week (using last 5 daily bars)
    pivot_points = np.full(len(daily_close), np.nan)
    for i in range(5, len(daily_close)):
        # Use previous week (5 trading days) high/low/close
        week_high = np.max(daily_high[i-5:i])
        week_low = np.min(daily_low[i-5:i])
        week_close = daily_close[i-1]
        
        # Standard pivot calculation
        pivot = (week_high + week_low + week_close) / 3.0
        r1 = 2 * pivot - week_low
        s1 = 2 * pivot - week_high
        r2 = pivot + (week_high - week_low)
        s2 = pivot - (week_high - week_low)
        r3 = week_high + 2 * (pivot - week_low)
        s3 = week_low - 2 * (week_high - pivot)
        
        # Store R2 and S2 for breakout confirmation
        pivot_points[i] = (r2, s2)
    
    # Extract R2 and S2 levels
    r2_levels = np.full(len(daily_close), np.nan)
    s2_levels = np.full(len(daily_close), np.nan)
    for i in range(len(daily_close)):
        if not np.isnan(pivot_points[i]):
            r2_levels[i] = pivot_points[i][0]
            s2_levels[i] = pivot_points[i][1]
    
    # Align pivot levels to 6h timeframe
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_levels)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_levels)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if close[i] <= donchian_low[i] or volume[i] < volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_high[i] or volume[i] < volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with pivot confirmation
            # Long: break above Donchian high AND above 1d R2 with volume
            if (close[i] > donchian_high[i-1] and 
                close[i] > r2_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low AND below 1d S2 with volume
            elif (close[i] < donchian_low[i-1] and 
                  close[i] < s2_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals
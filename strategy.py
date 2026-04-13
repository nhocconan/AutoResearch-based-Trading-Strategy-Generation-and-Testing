#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1d/1w trend filter and volume confirmation
# Strategy: Long when price closes above 1d high of previous 10 days and 1w close > 1w open (bullish week)
#           with volume > 1.3x 20-period average
#           Short when price closes below 1d low of previous 10 days and 1w close < 1w open (bearish week)
#           with volume > 1.3x 20-period average
# Uses multi-timeframe alignment: 1d for price channels, 1w for trend filter
# Volume surge confirms breakout strength in direction of higher timeframe trend
# Target: 15-35 total trades over 4 years (4-9/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for price channels (10-day high/low)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Calculate 10-day high/low channels
    high_10_1d = pd.Series(high_1d).rolling(window=10, min_periods=10).max().values
    low_10_1d = pd.Series(low_1d).rolling(window=10, min_periods=10).min().values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Weekly bullish/bearish filter
    weekly_bullish = close_1w > open_1w
    weekly_bearish = close_1w < open_1w
    
    # Volume average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    high_10_1d_aligned = align_htf_to_ltf(prices, df_1d, high_10_1d)
    low_10_1d_aligned = align_htf_to_ltf(prices, df_1d, low_10_1d)
    
    # Align 1w indicators to 4h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(high_10_1d_aligned[i]) or 
            np.isnan(low_10_1d_aligned[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or 
            np.isnan(weekly_bearish_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume surge condition
        volume_surge = volume[i] > 1.3 * vol_ma_20[i]
        
        # Breakout conditions with weekly trend filter
        long_breakout = close[i] > high_10_1d_aligned[i]
        short_breakout = close[i] < low_10_1d_aligned[i]
        
        # Entry logic: breakout in direction of weekly trend with volume
        long_entry = long_breakout and weekly_bullish_aligned[i] > 0.5 and volume_surge
        short_entry = short_breakout and weekly_bearish_aligned[i] > 0.5 and volume_surge
        
        # Exit conditions: opposite breakout or loss of weekly trend alignment
        exit_long = position == 1 and (short_breakout or weekly_bullish_aligned[i] < 0.5)
        exit_short = position == -1 and (long_breakout or weekly_bearish_aligned[i] < 0.5)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_1w_trend_filter_volume_breakout_v1"
timeframe = "4h"
leverage = 1.0
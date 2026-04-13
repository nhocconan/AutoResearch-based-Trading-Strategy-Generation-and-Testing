#!/usr/bin/env python3
"""
1d Donchian Breakout with Weekly EMA Trend Filter and Volume Confirmation.
Trades daily breakouts above/below Donchian channels (20-period) confirmed by volume spikes,
only in trending markets (weekly EMA cross) to avoid false breakouts in ranging conditions.
Designed for 1d timeframe to target 30-100 total trades over 4 years (7-25/year).
Works in both bull and bear markets by following the weekly trend direction.
"""

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
    
    # Get daily data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian Channel (20-period)
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Breakout conditions
    breakout_up = high_1d > upper_20
    breakout_down = low_1d < lower_20
    
    # Align signals to daily timeframe
    breakout_up_aligned = align_htf_to_ltf(prices, df_1d, breakout_up.astype(float))
    breakout_down_aligned = align_htf_to_ltf(prices, df_1d, breakout_down.astype(float))
    
    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA (21-period) for trend filter
    ema_21 = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Trend: price above EMA = uptrend, below EMA = downtrend
    uptrend = close_1w > ema_21
    downtrend = close_1w < ema_21
    
    # Align trend signals to daily timeframe
    uptrend_aligned = align_htf_to_ltf(prices, df_1w, uptrend.astype(float))
    downtrend_aligned = align_htf_to_ltf(prices, df_1w, downtrend.astype(float))
    
    # Get daily volume for confirmation
    volume_1d = df_1d['volume'].values
    
    # Volume spike: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma_20 * 2.0)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(breakout_up_aligned[i]) or 
            np.isnan(breakout_down_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or 
            np.isnan(uptrend_aligned[i]) or 
            np.isnan(downtrend_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Donchian breakout + volume spike + weekly trend
        long_entry = (breakout_up_aligned[i] > 0.5 and 
                      vol_spike_aligned[i] > 0.5 and 
                      uptrend_aligned[i] > 0.5)
        short_entry = (breakout_down_aligned[i] > 0.5 and 
                       vol_spike_aligned[i] > 0.5 and 
                       downtrend_aligned[i] > 0.5)
        
        # Exit when price returns to middle of Donchian channel
        middle = (upper_20 + lower_20) / 2
        middle_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(close_1d, middle))
        
        exit_long = position == 1 and close[i] <= middle_aligned[i]
        exit_short = position == -1 and close[i] >= middle_aligned[i]
        
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

name = "1d_donchian_breakout_weekly_trend"
timeframe = "1d"
leverage = 1.0
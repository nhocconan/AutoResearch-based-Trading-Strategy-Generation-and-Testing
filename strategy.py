#!/usr/bin/env python3
"""
12h Donchian Breakout + Weekly Trend + Volume Spike
Uses weekly Donchian channels (20-period) for trend direction and daily Donchian breakout for entry.
Volume spike confirms breakout strength. Designed for low frequency with strong trend-following edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def calculate_donchian(high, low, period=20):
    """Calculate Donchian channels: upper = max(high, period), lower = min(low, period)"""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=period, min_periods=period).max().values
    lower = low_series.rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend direction (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian for trend direction
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_upper, weekly_lower = calculate_donchian(weekly_high, weekly_low, 20)
    
    # Align weekly trend to 12h (wait for weekly bar to close)
    weekly_trend_up = align_ltf_to_htf(prices, df_1w, weekly_upper)
    weekly_trend_down = align_ltf_to_htf(prices, df_1w, weekly_lower)
    
    # Get daily data for entry signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily Donchian for entry
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_upper, daily_lower = calculate_donchian(daily_high, daily_low, 20)
    
    # Align daily levels to 12h (wait for daily bar to close)
    daily_upper_aligned = align_ltf_to_htf(prices, df_1d, daily_upper)
    daily_lower_aligned = align_ltf_to_htf(prices, df_1d, daily_lower)
    
    # Volume spike detection (2x 4-period average)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_trend_up[i]) or np.isnan(weekly_trend_down[i]) or 
            np.isnan(daily_upper_aligned[i]) or np.isnan(daily_lower_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Determine trend from weekly Donchian
        # Uptrend: price above weekly upper band
        # Downtrend: price below weekly lower band
        # No trend: between bands
        uptrend = price > weekly_trend_up[i]
        downtrend = price < weekly_trend_down[i]
        
        if position == 0:
            # Long: price breaks above daily upper with volume spike and weekly uptrend
            if (price > daily_upper_aligned[i] and 
                volume_spike[i] and uptrend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily lower with volume spike and weekly downtrend
            elif (price < daily_lower_aligned[i] and 
                  volume_spike[i] and downtrend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price crosses below daily lower (trend reversal)
            if price < daily_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price crosses above daily upper (trend reversal)
            if price > daily_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian_Breakout_WeeklyTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0
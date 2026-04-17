#!/usr/bin/env python3
"""
1d_Donchian20_WeeklyTrend_Filter
Hypothesis: Daily Donchian(20) breakout with weekly trend filter (price > weekly SMA40) captures strong trends while avoiding counter-trend whipsaws. Works in bull (captures rallies) and bear (avoids false longs in downtrends) by requiring alignment with weekly trend. Volume confirmation ensures breakout legitimacy. Low trade frequency (~10-20/year) minimizes fee drag.
"""

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
    
    # Calculate Donchian channels (20-period)
    def donchian_channels(high, low, window):
        upper = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    dc_upper, dc_lower = donchian_channels(high, low, 20)
    
    # Volume confirmation (20-period average)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly SMA40 for trend filter
    sma40_weekly = pd.Series(close_weekly).rolling(window=40, min_periods=40).mean().values
    
    # Align weekly SMA40 to daily timeframe
    sma40_weekly_aligned = align_htf_to_ltf(prices, df_weekly, sma40_weekly)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 20, 40)  # Donchian, volume MA20, weekly SMA40
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(dc_upper[i]) or 
            np.isnan(dc_lower[i]) or 
            np.isnan(volume_ma20[i]) or 
            np.isnan(sma40_weekly_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Donchian breakout conditions
        breakout_up = close[i] > dc_upper[i]
        breakout_down = close[i] < dc_lower[i]
        
        # Weekly trend filter: price above/below weekly SMA40
        weekly_uptrend = close[i] > sma40_weekly_aligned[i]
        weekly_downtrend = close[i] < sma40_weekly_aligned[i]
        
        if position == 0:
            # Long: upward breakout + volume filter + weekly uptrend
            if breakout_up and volume_filter and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: downward breakout + volume filter + weekly downtrend
            elif breakout_down and volume_filter and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price touches or goes below weekly SMA40 (trailing stop)
            if close[i] <= sma40_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price touches or goes above weekly SMA40 (trailing stop)
            if close[i] >= sma40_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_WeeklyTrend_Filter"
timeframe = "1d"
leverage = 1.0
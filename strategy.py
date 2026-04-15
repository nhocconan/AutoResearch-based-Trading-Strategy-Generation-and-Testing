#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 350:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for context
    weekly = get_htf_data(prices, '1w')
    weekly_high = weekly['high'].values
    weekly_low = weekly['low'].values
    weekly_close = weekly['close'].values
    
    # Calculate weekly pivot points
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly levels to 12h timeframe (wait for weekly close)
    weekly_pivot_12h = align_htf_to_ltf(prices, weekly, weekly_pivot)
    weekly_r1_12h = align_htf_to_ltf(prices, weekly, weekly_r1)
    weekly_s1_12h = align_htf_to_ltf(prices, weekly, weekly_s1)
    
    # Get daily data for volume filter
    daily = get_htf_data(prices, '1d')
    daily_volume = daily['volume'].values
    daily_volume_ma = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    daily_volume_ma_12h = align_htf_to_ltf(prices, daily, daily_volume_ma)
    
    # Volume filter: current volume > 1.5x daily average
    volume_filter = volume > (1.5 * daily_volume_ma_12h)
    
    # Trend filter: price above/below weekly pivot
    trend_filter = np.abs(close - weekly_pivot_12h) / weekly_pivot_12h > 0.005
    
    signals = np.zeros(n)
    
    for i in range(350, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_pivot_12h[i]) or np.isnan(weekly_r1_12h[i]) or 
            np.isnan(weekly_s1_12h[i]) or np.isnan(daily_volume_ma_12h[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when filters pass
        if volume_filter[i] and trend_filter[i]:
            # Long: break above weekly R1 with volume
            if close[i] > weekly_r1_12h[i]:
                signals[i] = 0.25
            # Short: break below weekly S1 with volume
            elif close[i] < weekly_s1_12h[i]:
                signals[i] = -0.25
            else:
                signals[i] = signals[i-1]
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_WeeklyPivot_R1_S1_Breakout_Volume_TrendFilter"
timeframe = "12h"
leverage = 1.0
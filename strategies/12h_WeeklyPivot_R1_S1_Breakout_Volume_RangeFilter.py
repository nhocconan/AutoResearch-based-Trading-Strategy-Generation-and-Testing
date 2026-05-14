#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly pivot levels
    daily = get_htf_data(prices, '1d')
    daily_high = daily['high'].values
    daily_low = daily['low'].values
    daily_close = daily['close'].values
    
    # Calculate weekly pivot levels from daily data
    weekly_high = pd.Series(daily_high).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(daily_low).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(daily_close).rolling(window=5, min_periods=5).last().values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly pivot levels to 12h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, daily, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, daily, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, daily, weekly_s1)
    
    # Volume filter: current 12h volume > 1.5x 20-period average volume
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Range filter: avoid trading when price is within 0.3% of weekly pivot
    price_to_pivot = np.abs(close - weekly_pivot_aligned) / weekly_pivot_aligned
    range_filter = price_to_pivot > 0.003
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when volume filter and range filter both pass
        if volume_filter[i] and range_filter[i]:
            # Long conditions: price breaks above weekly R1 with volume
            if close[i] > weekly_r1_aligned[i]:
                signals[i] = 0.25
            # Short conditions: price breaks below weekly S1 with volume
            elif close[i] < weekly_s1_aligned[i]:
                signals[i] = -0.25
            else:
                signals[i] = signals[i-1]
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_WeeklyPivot_R1_S1_Breakout_Volume_RangeFilter"
timeframe = "12h"
leverage = 1.0
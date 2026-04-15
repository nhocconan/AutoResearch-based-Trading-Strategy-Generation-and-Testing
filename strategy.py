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
    
    # Get daily data for weekly pivot calculation (last complete week)
    daily = get_htf_data(prices, '1d')
    daily_high = daily['high'].values
    daily_low = daily['low'].values
    daily_close = daily['close'].values
    
    # Calculate weekly pivot points using last week's daily data
    # We'll use the most recent complete week (last 5 trading days)
    # For simplicity, we'll calculate pivots on a rolling weekly basis
    # but we need to ensure we only use completed weeks
    
    # Create a weekly series by sampling daily data at weekly intervals
    # Since we don't have explicit week grouping, we'll use a rolling window
    # of 5 days and update only when we have a complete week
    # However, for pivot points, we typically use the prior week's data
    
    # Simpler approach: calculate daily pivots and use them as reference
    # but with weekly context - we'll use the pivot from 1 week ago
    
    # Calculate daily pivot points (standard floor trader pivots)
    daily_pivot = (daily_high + daily_low + daily_close) / 3.0
    daily_r1 = 2 * daily_pivot - daily_low
    daily_s1 = 2 * daily_pivot - daily_high
    daily_r2 = daily_pivot + (daily_high - daily_low)
    daily_s2 = daily_pivot - (daily_high - daily_low)
    
    # Align daily pivot levels to 6h timeframe (1 week delay for weekly context)
    # We want to use the pivot from 1 week ago, so we delay by 5 days (approx)
    # Since 1 week = 5 trading days, we delay by 5 periods in daily data
    pivot_weekly = np.roll(daily_pivot, 5)  # Shift by 5 days to get last week's pivot
    r1_weekly = np.roll(daily_r1, 5)
    s1_weekly = np.roll(daily_s1, 5)
    r2_weekly = np.roll(daily_r2, 5)
    s2_weekly = np.roll(daily_s2, 5)
    
    # Set first 5 values to NaN since we don't have prior week data
    pivot_weekly[:5] = np.nan
    r1_weekly[:5] = np.nan
    s1_weekly[:5] = np.nan
    r2_weekly[:5] = np.nan
    s2_weekly[:5] = np.nan
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, daily, pivot_weekly)
    r1_aligned = align_htf_to_ltf(prices, daily, r1_weekly)
    s1_aligned = align_htf_to_ltf(prices, daily, s1_weekly)
    r2_aligned = align_htf_to_ltf(prices, daily, r2_weekly)
    s2_aligned = align_htf_to_ltf(prices, daily, s2_weekly)
    
    # Volume filter: current 6h volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Trend filter: price above/below 20-period EMA for trend context
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema20[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when volume filter passes
        if volume_filter[i]:
            # Long conditions: price breaks above weekly R1 with volume and above EMA20
            if close[i] > r1_aligned[i] and close[i] > ema20[i]:
                signals[i] = 0.25
            # Long conditions: price bounces from weekly S1 with volume (above S1, below pivot)
            elif close[i] > s1_aligned[i] and close[i] < pivot_aligned[i] and close[i] > ema20[i]:
                signals[i] = 0.25
            # Short conditions: price breaks below weekly S1 with volume and below EMA20
            elif close[i] < s1_aligned[i] and close[i] < ema20[i]:
                signals[i] = -0.25
            # Short conditions: price rejected at weekly R1 with volume (below R1, above S1)
            elif close[i] < r1_aligned[i] and close[i] > s1_aligned[i] and close[i] < ema20[i]:
                signals[i] = -0.25
            else:
                signals[i] = signals[i-1]
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_WeeklyPivot_R1_S1_Breakout_Volume_TrendFilter"
timeframe = "6h"
leverage = 1.0
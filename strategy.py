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
    
    # Get daily data for pivot levels
    daily = get_htf_data(prices, '1d')
    daily_high = daily['high'].values
    daily_low = daily['low'].values
    daily_close = daily['close'].values
    
    # Calculate weekly data for higher timeframe bias
    weekly = get_htf_data(prices, '1w')
    weekly_high = weekly['high'].values
    weekly_low = weekly['low'].values
    weekly_close = weekly['close'].values
    
    # Calculate daily pivot levels (classic floor trader pivots)
    pivot_d = (daily_high + daily_low + daily_close) / 3.0
    r1_d = 2 * pivot_d - daily_low
    s1_d = 2 * pivot_d - daily_high
    r2_d = pivot_d + (daily_high - daily_low)
    s2_d = pivot_d - (daily_high - daily_low)
    
    # Calculate weekly pivot levels
    pivot_w = (weekly_high + weekly_low + weekly_close) / 3.0
    r1_w = 2 * pivot_w - weekly_low
    s1_w = 2 * pivot_w - weekly_high
    
    # Align daily pivot levels to 6h timeframe
    pivot_d_aligned = align_htf_to_ltf(prices, daily, pivot_d)
    r1_d_aligned = align_htf_to_ltf(prices, daily, r1_d)
    s1_d_aligned = align_htf_to_ltf(prices, daily, s1_d)
    r2_d_aligned = align_htf_to_ltf(prices, daily, r2_d)
    s2_d_aligned = align_htf_to_ltf(prices, daily, s2_d)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, weekly, pivot_w)
    r1_w_aligned = align_htf_to_ltf(prices, weekly, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, weekly, s1_w)
    
    # Volume filter: current 6h volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Trend filter: price above/below weekly pivot
    trend_filter_up = close > pivot_w_aligned
    trend_filter_down = close < pivot_w_aligned
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_d_aligned[i]) or np.isnan(r1_d_aligned[i]) or 
            np.isnan(s1_d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(pivot_w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when volume filter passes
        if volume_filter[i]:
            # Long conditions: price breaks above R1 with volume and above weekly pivot
            if close[i] > r1_d_aligned[i] and trend_filter_up[i]:
                signals[i] = 0.25
            # Long conditions: price bounces from S1 with volume and above weekly pivot
            elif close[i] > s1_d_aligned[i] and close[i] < s2_d_aligned[i] and trend_filter_up[i]:
                signals[i] = 0.25
            # Short conditions: price breaks below S1 with volume and below weekly pivot
            elif close[i] < s1_d_aligned[i] and trend_filter_down[i]:
                signals[i] = -0.25
            # Short conditions: price rejected at R1 with volume and below weekly pivot
            elif close[i] < r1_d_aligned[i] and close[i] > r2_d_aligned[i] and trend_filter_down[i]:
                signals[i] = -0.25
            else:
                signals[i] = signals[i-1]
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_Pivot_R1_S1_Breakout_Volume_WeeklyTrendFilter"
timeframe = "6h"
leverage = 1.0
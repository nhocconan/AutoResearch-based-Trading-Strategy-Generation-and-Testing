#!/usr/bin/env python3
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1w = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align daily pivot levels to 4h timeframe (use previous day's levels)
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average (20 periods = 10 days at 4h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need sufficient data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_4h[i]) or np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or
            np.isnan(volume_ma20[i]) or np.isnan(trend_1w[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        if position == 0:
            # Long: price breaks above R1 with volume AND above weekly trend
            if (close[i] > r1_4h[i] and volume_filter and close[i] > trend_1w[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume AND below weekly trend
            elif (close[i] < s1_4h[i] and volume_filter and close[i] < trend_1w[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below S1 or drops below weekly trend
            if close[i] < s1_4h[i] or close[i] < trend_1w[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above R1 or rises above weekly trend
            if close[i] > r1_4h[i] or close[i] > trend_1w[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WeeklyTrend_PivotBreakout_Volume"
timeframe = "4h"
leverage = 1.0
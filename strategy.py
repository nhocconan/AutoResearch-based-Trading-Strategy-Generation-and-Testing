#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_DailyBreakout_TrendFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for daily pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Get 1w data for weekly pivot direction (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate daily pivot points (using previous day's OHLC)
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    # Standard pivot point
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    # Support and resistance levels
    r1_1d = 2 * pivot_1d - prev_low_1d
    s1_1d = 2 * pivot_1d - prev_high_1d
    r2_1d = pivot_1d + (prev_high_1d - prev_low_1d)
    s2_1d = pivot_1d - (prev_high_1d - prev_low_1d)
    
    # Calculate weekly pivot for trend direction (using previous week's OHLC)
    prev_close_1w = df_1w['close'].shift(1).values
    prev_high_1w = df_1w['high'].shift(1).values
    prev_low_1w = df_1w['low'].shift(1).values
    
    # Weekly pivot point
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    # Weekly trend: price above/below weekly pivot
    trend_up_1w = prev_close_1w > pivot_1w
    trend_down_1w = prev_close_1w < pivot_1w
    
    # Align all levels to 6h timeframe
    pivot_1d_6h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_6h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_6h = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_6h = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_6h = align_htf_to_ltf(prices, df_1d, s2_1d)
    trend_up_1w_6h = align_htf_to_ltf(prices, df_1w, trend_up_1w.astype(float))
    trend_down_1w_6h = align_htf_to_ltf(prices, df_1w, trend_down_1w.astype(float))
    
    # Volume filter: above 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_1d_6h[i]) or np.isnan(r1_1d_6h[i]) or np.isnan(s1_1d_6h[i]) or
            np.isnan(r2_1d_6h[i]) or np.isnan(s2_1d_6h[i]) or np.isnan(trend_up_1w_6h[i]) or
            np.isnan(trend_down_1w_6h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        # Session filter: 08-20 UTC (reduce noise trades)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long entry: price breaks above R1 with weekly uptrend
            if (close[i] > r1_1d_6h[i] and 
                trend_up_1w_6h[i] > 0.5 and  # Weekly uptrend
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 with weekly downtrend
            elif (close[i] < s1_1d_6h[i] and 
                  trend_down_1w_6h[i] > 0.5 and  # Weekly downtrend
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back to pivot (mean reversion to pivot)
            if close[i] < pivot_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back to pivot (mean reversion to pivot)
            if close[i] > pivot_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_DailyBreakout_TrendFilter_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Previous weekly OHLC for pivot calculation
    prev_high_1w = df_1w['high'].shift(1).values
    prev_low_1w = df_1w['low'].shift(1).values
    prev_close_1w = df_1w['close'].shift(1).values
    
    # Weekly pivot point
    weekly_pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    weekly_range_1w = prev_high_1w - prev_low_1w
    # Weekly resistance/support levels
    weekly_r1_1w = weekly_pivot_1w + weekly_range_1w * 0.382
    weekly_s1_1w = weekly_pivot_1w - weekly_range_1w * 0.382
    weekly_r2_1w = weekly_pivot_1w + weekly_range_1w * 0.618
    weekly_s2_1w = weekly_pivot_1w - weekly_range_1w * 0.618
    
    # Align weekly levels to 6h
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1w, weekly_pivot_1w)
    weekly_r1_6h = align_htf_to_ltf(prices, df_1w, weekly_r1_1w)
    weekly_s1_6h = align_htf_to_ltf(prices, df_1w, weekly_s1_1w)
    weekly_r2_6h = align_htf_to_ltf(prices, df_1w, weekly_r2_1w)
    weekly_s2_6h = align_htf_to_ltf(prices, df_1w, weekly_s2_1w)
    
    # Daily trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: above 1.8x 24-period average (24*6h = 6 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_r1_6h[i]) or np.isnan(weekly_s1_6h[i]) or 
            np.isnan(weekly_r2_6h[i]) or np.isnan(weekly_s2_6h[i]) or 
            np.isnan(ema_50_6h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.8 * vol_ma[i]  # Volume confirmation
        
        # Session filter: 08-20 UTC (reduce noise trades)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long breakout: price breaks above weekly R2 with daily uptrend
            if (close[i] > weekly_r2_6h[i] and 
                close[i] > ema_50_6h[i] and  # daily uptrend
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below weekly S2 with daily downtrend
            elif (close[i] < weekly_s2_6h[i] and 
                  close[i] < ema_50_6h[i] and  # daily downtrend
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below weekly pivot
            if close[i] < weekly_pivot_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above weekly pivot
            if close[i] > weekly_pivot_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyPivot_DailyBreakout_TrendFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 100:
        return np.zeros(n)
    
    # Previous week's OHLC (for weekly pivot)
    prev_close_1w = df_1w['close'].shift(1).values
    prev_high_1w = df_1w['high'].shift(1).values
    prev_low_1w = df_1w['low'].shift(1).values
    
    # Calculate weekly pivot
    weekly_pivot = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    weekly_range = prev_high_1w - prev_low_1w
    weekly_r1 = weekly_pivot + weekly_range * 1.1 / 4  # R1 resistance
    weekly_s1 = weekly_pivot - weekly_range * 1.1 / 4  # S1 support
    
    # Align weekly levels to daily
    weekly_pivot_d = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_d = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_d = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily volume filter: above 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_r1_d[i]) or np.isnan(weekly_s1_d[i]) or 
            np.isnan(ema_50_1d[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma[i]  # Volume confirmation
        
        if position == 0:
            # Long breakout: price breaks above weekly R1 with weekly uptrend
            if (close[i] > weekly_r1_d[i] and 
                close[i] > ema_50_1d[i] and  # weekly uptrend
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below weekly S1 with weekly downtrend
            elif (close[i] < weekly_s1_d[i] and 
                  close[i] < ema_50_1d[i] and  # weekly downtrend
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below weekly pivot
            if close[i] < weekly_pivot_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above weekly pivot
            if close[i] > weekly_pivot_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
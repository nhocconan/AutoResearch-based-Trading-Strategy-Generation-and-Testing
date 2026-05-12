#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

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
    
    # Daily Camarilla levels (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # Camarilla calculation: based on previous day's range
    range_1d = high_1d - low_1d
    close_prev = np.roll(close_1d, 1)
    close_prev[0] = close_1d[0]  # first value
    high_prev = np.roll(high_1d, 1)
    high_prev[0] = high_1d[0]
    low_prev = np.roll(low_1d, 1)
    low_prev[0] = low_1d[0]
    range_prev = high_prev - low_prev
    
    # Camarilla levels: based on previous day
    pivot = (high_prev + low_prev + close_prev) / 3.0
    r1 = close_prev + (range_prev * 1.1 / 12)
    s1 = close_prev - (range_prev * 1.1 / 12)
    r2 = close_prev + (range_prev * 1.1 / 6)
    s2 = close_prev - (range_prev * 1.1 / 6)
    r3 = close_prev + (range_prev * 1.1 / 4)
    s3 = close_prev - (range_prev * 1.1 / 4)
    r4 = close_prev + (range_prev * 1.1 / 2)
    s4 = close_prev - (range_prev * 1.1 / 2)
    
    # Weekly trend filter (1w EMA34)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_trend_up = close_1w > ema34_1w
    weekly_trend_down = close_1w < ema34_1w
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up)
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down)
    
    # Align Camarilla levels (based on previous day, so align as-is)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume filter: 4h volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # ensure weekly EMA has enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(weekly_trend_up_aligned[i]) or np.isnan(weekly_trend_down_aligned[i]) or np.isnan(volume_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: weekly trend up + price breaks above R1 + volume filter
            if (weekly_trend_up_aligned[i] and 
                close[i] > r1_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: weekly trend down + price breaks below S1 + volume filter
            elif (weekly_trend_down_aligned[i] and 
                  close[i] < s1_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below S1 OR weekly trend changes to down
            if close[i] < s1_aligned[i] or weekly_trend_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above R1 OR weekly trend changes to up
            if close[i] > r1_aligned[i] or weekly_trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
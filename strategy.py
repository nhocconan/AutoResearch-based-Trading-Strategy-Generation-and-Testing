#!/usr/bin/env python3
name = "6h_WeeklyPivot_DonchianBreakout_TrendFilter"
timeframe = "6h"
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
    
    # Weekly data for pivot points and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly pivot point calculation (based on prior week)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate pivots for each week
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    
    # Align weekly pivots to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Weekly trend filter: price above/below weekly pivot
    weekly_trend_up = weekly_close > pivot
    weekly_trend_down = weekly_close < pivot
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up)
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down)
    
    # Daily Donchian channels for breakout signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 20-day Donchian channels
    high_20d = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    low_20d = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian to 6h
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, high_20d)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, low_20d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Time-based session filter: active during major sessions
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if weekly data not ready
        if np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if daily Donchian not ready
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Session filter: London/NY overlap (08-16 UTC) and Asia (00-08 UTC)
        hour = hours[i]
        in_session = ((0 <= hour <= 8) or (8 <= hour <= 16))
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long conditions: price breaks above weekly R3 with weekly uptrend and Donchian breakout
            if (high[i] > r3_aligned[i] and 
                close[i] > r3_aligned[i] and
                weekly_trend_up_aligned[i] and  # weekly uptrend
                close[i] > donchian_high_aligned[i]):  # Donchian breakout
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below weekly S3 with weekly downtrend and Donchian breakout
            elif (low[i] < s3_aligned[i] and 
                  close[i] < s3_aligned[i] and
                  weekly_trend_down_aligned[i] and  # weekly downtrend
                  close[i] < donchian_low_aligned[i]):  # Donchian breakdown
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price breaks below weekly S3 or weekly trend turns down
            if (low[i] < s3_aligned[i] or 
                not weekly_trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price breaks above weekly R3 or weekly trend turns up
            if (high[i] > r3_aligned[i] or 
                weekly_trend_down_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
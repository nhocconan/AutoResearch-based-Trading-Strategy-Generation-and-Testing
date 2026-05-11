#!/usr/bin/env python3
name = "6h_Donchian20_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly pivot from previous week: (H + L + C) / 3
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_pivot_aligned = align_ltf_to_htf(prices, df_1w, weekly_pivot)
    
    # Daily trend: price above/below 20-day EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_ltf_to_htf(prices, df_1d, ema_20_1d)
    daily_trend_up = close > ema_20_1d_aligned
    
    # Donchian(20) on 6h: upper/lower band of last 20 bars
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(n):
        if i >= lookback - 1:
            start = i - lookback + 1
            highest_high[i] = np.max(high[start:i+1])
            lowest_low[i] = np.min(low[start:i+1])
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    volume_filter = volume > 1.5 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 19)  # Need enough data for Donchian and volume
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(ema_20_1d_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close above Donchian upper + above weekly pivot + daily uptrend + volume
            if (close[i] > highest_high[i] and 
                close[i] > weekly_pivot_aligned[i] and 
                daily_trend_up[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close below Donchian lower + below weekly pivot + daily downtrend + volume
            elif (close[i] < lowest_low[i] and 
                  close[i] < weekly_pivot_aligned[i] and 
                  not daily_trend_up[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close below Donchian lower or below weekly pivot
            if close[i] < lowest_low[i] or close[i] < weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close above Donchian upper or above weekly pivot
            if close[i] > highest_high[i] or close[i] > weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
name = "6h_Donchian20_Breakout_1dTrend_VolumeFilter_WeeklyPivot"
timeframe = "6h"
leverage = 1.0

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
    
    # Load 1d data for trend filter and weekly pivot
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Weekly pivot: use 1d data to calculate weekly pivot (approx using 5-day window)
    # Pivot = (High + Low + Close) / 3 for the week (using last 5 days)
    # We'll use the pivot from 5 days ago to avoid look-ahead
    lookback = 5
    weekly_high = np.zeros_like(high_1d)
    weekly_low = np.zeros_like(low_1d)
    weekly_close = np.zeros_like(close_1d)
    
    for i in range(len(close_1d)):
        if i >= lookback:
            weekly_high[i] = np.max(high_1d[i-lookback:i])
            weekly_low[i] = np.min(low_1d[i-lookback:i])
            weekly_close[i] = close_1d[i-1]  # previous day's close
        else:
            weekly_high[i] = np.nan
            weekly_low[i] = np.nan
            weekly_close[i] = np.nan
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    # Donchian channels (20-period) on 6h data
    lookback_dc = 20
    dc_high = np.zeros_like(high)
    dc_low = np.zeros_like(low)
    
    for i in range(len(close)):
        if i >= lookback_dc:
            dc_high[i] = np.max(high[i-lookback_dc:i])
            dc_low[i] = np.min(low[i-lookback_dc:i])
        else:
            dc_high[i] = np.nan
            dc_low[i] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, lookback_dc, lookback)  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: breakout above Donchian high + above 1d EMA50 + above weekly pivot + volume filter
            if high[i] > dc_high[i] and close[i] > ema_50_1d_aligned[i] and close[i] > weekly_pivot_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian low + below 1d EMA50 + below weekly pivot + volume filter
            elif low[i] < dc_low[i] and close[i] < ema_50_1d_aligned[i] and close[i] < weekly_pivot_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: breakdown below Donchian low or below 1d EMA50
            if low[i] < dc_low[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: breakout above Donchian high or above 1d EMA50
            if high[i] > dc_high[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
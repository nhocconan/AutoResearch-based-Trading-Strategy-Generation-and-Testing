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
    
    # Get daily data for weekly pivot calculation (1d timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot from daily data (last 5 days)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot using last 5 daily bars
    weekly_pivot = np.full(len(high_1d), np.nan)
    for i in range(4, len(high_1d)):  # Need at least 5 days
        week_high = np.max(high_1d[i-4:i+1])
        week_low = np.min(low_1d[i-4:i+1])
        week_close = close_1d[i]
        weekly_pivot[i] = (week_high + week_low + week_close) / 3
    
    # Align weekly pivot to 12h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Calculate 20-period Donchian channels directly on 12h data
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        highest_high[i] = np.max(high[i-lookback:i])
        lowest_low[i] = np.min(low[i-lookback:i])
    
    # 20-period average volume for spike detection
    vol_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period: need at least 20 for Donchian, 20 for volume, 5 for weekly pivot
    start_idx = max(lookback, vol_period, 5)
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Determine trend from weekly pivot
        bullish = price > weekly_pivot_aligned[i]
        bearish = price < weekly_pivot_aligned[i]
        
        # Volume confirmation: spike > 2.0x average
        volume_confirmation = vol_ratio > 2.0
        
        if position == 0:
            # Long breakout: price breaks above Donchian high in bullish trend with volume
            if bullish and price > highest_high[i] and volume_confirmation:
                signals[i] = size
                position = 1
            # Short breakdown: price breaks below Donchian low in bearish trend with volume
            elif bearish and price < lowest_low[i] and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below Donchian low or trend turns bearish
            if price < lowest_low[i] or bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price breaks above Donchian high or trend turns bullish
            if price > highest_high[i] or bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian20_WeeklyPivot_Trend_Volume"
timeframe = "12h"
leverage = 1.0
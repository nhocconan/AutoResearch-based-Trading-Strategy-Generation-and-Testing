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
    
    # Get weekly data for monthly pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate monthly pivot from weekly data (last 4 weeks)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate monthly pivot using last 4 weekly bars
    monthly_pivot = np.full(len(high_1w), np.nan)
    for i in range(3, len(high_1w)):  # Need at least 4 weeks
        week_high = np.max(high_1w[i-3:i+1])
        week_low = np.min(low_1w[i-3:i+1])
        week_close = close_1w[i]
        monthly_pivot[i] = (week_high + week_low + week_close) / 3
    
    # Align monthly pivot to daily timeframe
    monthly_pivot_aligned = align_htf_to_ltf(prices, df_1w, monthly_pivot)
    
    # Calculate 20-period Donchian channels directly on daily data
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
    
    # Warmup period: need at least 20 for Donchian, 20 for volume, 4 for monthly pivot
    start_idx = max(lookback, vol_period, 3)
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(monthly_pivot_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Determine trend from monthly pivot
        bullish = price > monthly_pivot_aligned[i]
        bearish = price < monthly_pivot_aligned[i]
        
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

name = "1d_Donchian20_MonthlyPivot_Trend_Volume"
timeframe = "1d"
leverage = 1.0
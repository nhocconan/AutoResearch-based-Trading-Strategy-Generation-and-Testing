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
    
    # Get weekly data for trend filter (primary: daily)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 50-week EMA for trend filter
    ema_period = 50
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_period:
        ema_1w[ema_period - 1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * (2 / (ema_period + 1)) + 
                         ema_1w[i-1] * (1 - (2 / (ema_period + 1))))
    
    # Align weekly EMA to daily timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Get daily data for signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day Donchian channels
    highest_high = np.full(len(high_1d), np.nan)
    lowest_low = np.full(len(low_1d), np.nan)
    
    for i in range(19, len(high_1d)):
        highest_high[i] = np.max(high_1d[i-19:i+1])
        lowest_low[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate 20-day average volume
    avg_volume = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        avg_volume[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align Donchian levels and average volume to daily timeframe
    highest_high_aligned = align_htf_to_ltf(prices, df_1d, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_1d, lowest_low)
    avg_volume_aligned = align_htf_to_ltf(prices, df_1d, avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian channels and volume average
    start_idx = 19
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high_aligned[i]) or np.isnan(lowest_low_aligned[i]) or 
            np.isnan(avg_volume_aligned[i]) or np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume_aligned[i]
        upper_channel = highest_high_aligned[i]
        lower_channel = lowest_low_aligned[i]
        weekly_trend = ema_1w_aligned[i]
        
        if position == 0:
            # Long: Break above upper Donchian with volume confirmation and weekly uptrend
            if (price > upper_channel and vol > 1.5 * avg_vol and price > weekly_trend):
                signals[i] = size
                position = 1
            # Short: Break below lower Donchian with volume confirmation and weekly downtrend
            elif (price < lower_channel and vol > 1.5 * avg_vol and price < weekly_trend):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price returns to middle of channel or weekly trend fails
            if price < (upper_channel + lower_channel) / 2 or price < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Price returns to middle of channel or weekly trend fails
            if price > (upper_channel + lower_channel) / 2 or price > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1D_Donchian_Breakout_WeeklyTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0
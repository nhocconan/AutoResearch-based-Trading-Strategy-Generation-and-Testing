#!/usr/bin/env python3
name = "1D_WeeklyTrend_DailyBreakout_12hVolume"
timeframe = "1d"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA10 for trend filter
    ema10_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 10:
        ema10_1w[9] = np.mean(close_1w[0:10])
        for i in range(10, len(close_1w)):
            ema10_1w[i] = (close_1w[i] * 2 + ema10_1w[i-1] * 8) / 10
    
    # Align weekly EMA10 to daily timeframe
    ema10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema10_1w)
    
    # Get 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h volume EMA20
    vol_ema20_12h = np.full_like(volume_12h, np.nan)
    if len(volume_12h) >= 20:
        vol_ema20_12h[19] = np.mean(volume_12h[0:20])
        for i in range(20, len(volume_12h)):
            vol_ema20_12h[i] = (volume_12h[i] * 2 + vol_ema20_12h[i-1] * 18) / 20
    
    # Align 12h volume EMA20 to daily timeframe
    vol_ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ema20_12h)
    
    # Calculate daily Donchian breakout levels (20-period)
    highest_high_20d = np.full_like(close, np.nan)
    lowest_low_20d = np.full_like(close, np.nan)
    
    for i in range(n):
        if i >= 19:
            start_idx = i - 19
            highest_high_20d[i] = np.max(high[start_idx:i+1])
            lowest_low_20d[i] = np.min(low[start_idx:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(19, 9, 19)  # Need Donchian(20), weekly EMA10, 12h vol EMA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema10_1w_aligned[i]) or np.isnan(vol_ema20_12h_aligned[i]) or 
            np.isnan(highest_high_20d[i]) or np.isnan(lowest_low_20d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market conditions
        weekly_trend_up = close[i] > ema10_1w_aligned[i]
        volume_surge = volume[i] > vol_ema20_12h_aligned[i] * 1.5
        
        if position == 0:
            # Enter long: Uptrend + price breaks above 20-day high + volume surge
            if weekly_trend_up and close[i] > highest_high_20d[i] and volume_surge:
                signals[i] = 0.25
                position = 1
            # Enter short: Downtrend + price breaks below 20-day low + volume surge
            elif not weekly_trend_up and close[i] < lowest_low_20d[i] and volume_surge:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Trend turns down OR price breaks below 20-day low
            if not weekly_trend_up or close[i] < lowest_low_20d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Trend turns up OR price breaks above 20-day high
            if weekly_trend_up or close[i] > highest_high_20d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
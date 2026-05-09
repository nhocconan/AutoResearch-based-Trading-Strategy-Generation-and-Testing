#!/usr/bin/env python3
"""
1d_WeeklyTrend_DailyBreakout_12hVolume
Hypothesis: Trade with weekly trend on daily timeframe using Donchian breakouts,
filtered by 12h volume surge to ensure institutional participation.
Works in bull markets via trend-following breakouts and in bear markets via 
short-side breakdowns with volume confirmation. Targets 15-25 trades/year.
"""

name = "1d_WeeklyTrend_DailyBreakout_12hVolume"
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
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Calculate weekly EMA20 for trend filter
    ema20_weekly = np.full_like(close_weekly, np.nan)
    if len(close_weekly) >= 20:
        ema20_weekly[19] = np.mean(close_weekly[0:20])
        for i in range(20, len(close_weekly)):
            ema20_weekly[i] = (close_weekly[i] * 2 + ema20_weekly[i-1] * 18) / 20
    
    # Align weekly EMA20 to daily timeframe
    ema20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema20_weekly)
    
    # Get 12h data for volume filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h volume EMA20
    vol_ema_12h = np.full_like(volume_12h, np.nan)
    if len(volume_12h) >= 20:
        vol_ema_12h[19] = np.mean(volume_12h[0:20])
        for i in range(20, len(volume_12h)):
            vol_ema_12h[i] = (volume_12h[i] * 2 + vol_ema_12h[i-1] * 18) / 20
    
    # Align 12h volume EMA to daily timeframe
    vol_ema_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ema_12h)
    
    # Calculate daily Donchian channels (20-period)
    donchian_high = np.full_like(high, np.nan)
    donchian_low = np.full_like(low, np.nan)
    
    for i in range(n):
        if i >= 19:
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(19, 19)  # Need Donchian and aligned indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema20_weekly_aligned[i]) or np.isnan(vol_ema_12h_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine conditions
        weekly_uptrend = close[i] > ema20_weekly_aligned[i]
        volume_surge = volume[i] > vol_ema_12h_aligned[i] * 1.5
        
        if position == 0:
            # Enter long: weekly uptrend + price breaks above Donchian high + volume surge
            if weekly_uptrend and close[i] > donchian_high[i] and volume_surge:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly downtrend + price breaks below Donchian low + volume surge
            elif not weekly_uptrend and close[i] < donchian_low[i] and volume_surge:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly trend turns down OR price breaks below Donchian low
            if not weekly_uptrend or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly trend turns up OR price breaks above Donchian high
            if weekly_uptrend or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
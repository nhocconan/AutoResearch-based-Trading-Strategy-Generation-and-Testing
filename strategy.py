#!/usr/bin/env python3
# 12h_WeeklyDonchian_Breakout_Trend_1dVolume
# Hypothesis: Weekly Donchian breakouts combined with daily EMA trend and volume
# confirmation provide high-probability entries in both bull and bear markets.
# Weekly trend filter avoids whipsaws, while volume confirmation ensures institutional
# participation. Target: 15-30 trades/year to minimize fee drag.

name = "12h_WeeklyDonchian_Breakout_Trend_1dVolume"
timeframe = "12h"
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
    
    # Get weekly data for Donchian channels and trend
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Get daily data for EMA trend and volume
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Weekly Donchian channels (20 periods)
    weekly_high = df_weekly['high'].rolling(window=20, min_periods=20).max().values
    weekly_low = df_weekly['low'].rolling(window=20, min_periods=20).min().values
    weekly_high_aligned = align_htf_to_ltf(prices, df_weekly, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_weekly, weekly_low)
    
    # Daily EMA50 for trend filter
    ema_50_daily = pd.Series(df_daily['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_50_daily)
    
    # Daily volume confirmation (20-period MA)
    volume_daily = df_daily['volume'].rolling(window=20, min_periods=20).mean().values
    volume_daily_aligned = align_htf_to_ltf(prices, df_daily, volume_daily)
    
    # Daily volume ratio for entry confirmation
    volume_ratio = volume / volume_daily_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: weekly Donchian (20), daily EMA50 (50)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i]) or 
            np.isnan(ema_50_daily_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price relative to daily EMA50
        uptrend = close[i] > ema_50_daily_aligned[i]
        downtrend = close[i] < ema_50_daily_aligned[i]
        
        # Volume confirmation: current volume > 1.5x daily average
        volume_confirm = volume_ratio[i] > 1.5
        
        if position == 0:
            # Long entry: uptrend + price breaks above weekly Donchian high + volume
            if uptrend and close[i] > weekly_high_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + price breaks below weekly Donchian low + volume
            elif downtrend and close[i] < weekly_low_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price re-enters Donchian range
            if not uptrend or close[i] < weekly_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price re-enters Donchian range
            if not downtrend or close[i] > weekly_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
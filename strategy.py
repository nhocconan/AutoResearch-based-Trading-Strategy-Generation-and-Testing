#!/usr/bin/env python3
"""
12h_1d_1w_Donchian_Breakout_v1
Hypothesis: Use weekly trend (price above/below weekly SMA10) as bias, daily Donchian(10) breakout for entry,
and 12h price action with volume confirmation. Only trade in direction of weekly trend.
Volume filter avoids false breakouts. Targets 15-25 trades/year to stay under fee drag.
Works in bull (follow weekly trend) and bear (fade counter-trend moves at daily levels) by requiring weekly alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_Donchian_Breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # === WEEKLY TREND FILTER ===
    weekly_close = df_1w['close'].values
    weekly_sma10 = np.full(len(weekly_close), np.nan)
    if len(weekly_close) >= 10:
        for i in range(10, len(weekly_close)):
            weekly_sma10[i] = np.mean(weekly_close[i-10:i])
    weekly_sma10_12h = align_htf_to_ltf(prices, df_1w, weekly_sma10)
    
    # === DAILY DONCHIAN CHANNELS (10-period) ===
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    upper_channel = np.full(len(daily_high), np.nan)
    lower_channel = np.full(len(daily_low), np.nan)
    
    if len(daily_high) >= 10:
        for i in range(10, len(daily_high)):
            upper_channel[i] = np.max(daily_high[i-10:i])
            lower_channel[i] = np.min(daily_low[i-10:i])
    
    upper_channel_12h = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_channel_12h = align_htf_to_ltf(prices, df_1d, lower_channel)
    
    # === VOLUME FILTER (20-period average) ===
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        vol_sum = np.sum(volume[:20])
        vol_ma[19] = vol_sum / 20
        for i in range(20, n):
            vol_sum = vol_sum - volume[i-20] + volume[i]
            vol_ma[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(upper_channel_12h[i]) or np.isnan(lower_channel_12h[i]) or 
            np.isnan(weekly_sma10_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = volume[i] > vol_ma[i] * 1.5
        
        # Weekly trend: price above/below weekly SMA10
        weekly_trend_up = close[i] > weekly_sma10_12h[i]
        
        # Daily breakout conditions
        breakout_up = high[i] > upper_channel_12h[i] and vol_confirm
        breakout_down = low[i] < lower_channel_12h[i] and vol_confirm
        
        # Entry logic: only trade in direction of weekly trend
        long_entry = breakout_up and weekly_trend_up
        short_entry = breakout_down and not weekly_trend_up
        
        # Exit logic: reverse signal or price returns to opposite channel
        long_exit = not breakout_up or low[i] < lower_channel_12h[i]
        short_exit = not breakout_down or high[i] > upper_channel_12h[i]
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
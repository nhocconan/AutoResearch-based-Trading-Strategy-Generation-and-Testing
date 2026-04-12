#!/usr/bin/env python3
"""
12h_1d_1w_Camarilla_Trend_Rotation_v1
Hypothesis: Use weekly pivot direction to set bias, daily Camarilla (R3/S3) for entry,
and 12h price action with volume confirmation. Only trade in direction of weekly trend
(price above/below weekly pivot). Weekly trend filter reduces whipsaws in ranging markets.
Targets 12-25 trades/year to stay under fee drag. Works in bull (follow weekly trend breaks)
and bear (fade counter-trend moves at daily levels) by requiring weekly alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_Camarilla_Trend_Rotation_v1"
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
    
    # Daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === WEEKLY TREND FILTER ===
    # Weekly pivot (using prior week)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    prev_weekly_high = np.roll(weekly_high, 1)
    prev_weekly_low = np.roll(weekly_low, 1)
    prev_weekly_close = np.roll(weekly_close, 1)
    prev_weekly_high[0] = weekly_high[0]
    prev_weekly_low[0] = weekly_low[0]
    prev_weekly_close[0] = weekly_close[0]
    
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3
    weekly_pivot_12h = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # === DAILY CAMARILLA LEVELS ===
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    prev_daily_high = np.roll(daily_high, 1)
    prev_daily_low = np.roll(daily_low, 1)
    prev_daily_close = np.roll(daily_close, 1)
    prev_daily_high[0] = daily_high[0]
    prev_daily_low[0] = daily_low[0]
    prev_daily_close[0] = daily_close[0]
    
    daily_pivot = (prev_daily_high + prev_daily_low + prev_daily_close) / 3
    daily_range = prev_daily_high - prev_daily_low
    
    # Camarilla R3/S3 (tighter bands for higher probability)
    R3 = daily_pivot + daily_range * 1.1 / 4
    S3 = daily_pivot - daily_range * 1.1 / 4
    
    R3_12h = align_htf_to_ltf(prices, df_1d, R3)
    S3_12h = align_htf_to_ltf(prices, df_1d, S3)
    
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
        if (np.isnan(R3_12h[i]) or np.isnan(S3_12h[i]) or 
            np.isnan(weekly_pivot_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.3x average
        vol_confirm = volume[i] > vol_ma[i] * 1.3
        
        # Weekly trend: price above/below weekly pivot
        weekly_trend_up = close[i] > weekly_pivot_12h[i]
        
        # Daily breakout conditions
        breakout_up = high[i] > R3_12h[i] and vol_confirm
        breakout_down = low[i] < S3_12h[i] and vol_confirm
        
        # Entry logic: only trade in direction of weekly trend
        long_entry = breakout_up and weekly_trend_up
        short_entry = breakout_down and not weekly_trend_up
        
        # Exit logic: reverse signal or price returns to daily pivot
        daily_pivot_12h = align_htf_to_ltf(prices, df_1d, daily_pivot)
        long_exit = not breakout_up or close[i] < daily_pivot_12h[i]
        short_exit = not breakout_down or close[i] > daily_pivot_12h[i]
        
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
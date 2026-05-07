#!/usr/bin/env python3
name = "6h_Donchian_20_WeeklyPivot_Direction_Volume"
timeframe = "6h"
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
    
    # Daily data for weekly pivot and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 7:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Weekly pivot from previous week (daily data)
    # Group into weeks ending Sunday
    weeks = []
    i = 0
    while i < len(df_1d):
        week_end = min(i + 7, len(df_1d))
        week_high = np.max(high_1d[i:week_end])
        week_low = np.min(low_1d[i:week_end])
        week_close = close_1d[week_end - 1]
        weeks.append((week_high, week_low, week_close))
        i = week_end
    
    if len(weeks) < 2:
        return np.zeros(n)
    
    # Previous week's data for pivot
    week_high_prev, week_low_prev, week_close_prev = weeks[-2]
    pivot = (week_high_prev + week_low_prev + week_close_prev) / 3.0
    r1 = 2 * pivot - week_low_prev
    s1 = 2 * pivot - week_high_prev
    r2 = pivot + (week_high_prev - week_low_prev)
    s2 = pivot - (week_high_prev - week_low_prev)
    
    # Align weekly pivot to 6h (assuming 4 6h bars per day)
    weeks_array = np.array(weeks)
    week_high_arr = weeks_array[:, 0]
    week_low_arr = weeks_array[:, 1]
    week_close_arr = weeks_array[:, 2]
    pivot_arr = (week_high_arr + week_low_arr + week_close_arr) / 3.0
    r1_arr = 2 * pivot_arr - week_low_arr
    s1_arr = 2 * pivot_arr - week_high_arr
    r2_arr = pivot_arr + (week_high_arr - week_low_arr)
    s2_arr = pivot_arr - (week_high_arr - week_low_arr)
    
    # Expand to daily then to 6h
    def expand_to_daily(arr):
        daily = np.repeat(arr, 7)[:len(df_1d)]
        return daily
    
    def expand_to_6h(arr):
        daily_expanded = expand_to_daily(arr)
        # Repeat each daily value 4 times for 6h (4*6h=24h)
        h6 = np.repeat(daily_expanded, 4)
        # Trim/pad to match prices length
        if len(h6) < n:
            h6 = np.pad(h6, (0, n - len(h6)), 'edge')
        else:
            h6 = h6[:n]
        return h6
    
    pivot_6h = expand_to_6h(pivot_arr)
    r1_6h = expand_to_6h(r1_arr)
    s1_6h = expand_to_6h(s1_arr)
    r2_6h = expand_to_6h(r2_arr)
    s2_6h = expand_to_6h(s2_arr)
    
    # Donchian channel (20-period) on 6h
    def donchian_channel(high, low, lookback):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(high, np.nan)
        for i in range(lookback - 1, len(high)):
            upper[i] = np.max(high[i - lookback + 1:i + 1])
            lower[i] = np.min(low[i - lookback + 1:i + 1])
        return upper, lower
    
    dc_upper, dc_lower = donchian_channel(high, low, 20)
    
    # Volume surge: current 6h volume > 2.0x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_surge = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 8  # ~2 days (8*6h) to reduce trade frequency
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or 
            np.isnan(s1_6h[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: break above Donchian upper with volume surge and price above weekly pivot
            if (close[i] > dc_upper[i] and 
                vol_surge[i] and 
                close[i] > pivot_6h[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: break below Donchian lower with volume surge and price below weekly pivot
            elif (close[i] < dc_lower[i] and 
                  vol_surge[i] and 
                  close[i] < pivot_6h[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: break below Donchian lower or price crosses below weekly pivot
            if close[i] < dc_lower[i] or close[i] < pivot_6h[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: break above Donchian upper or price crosses above weekly pivot
            if close[i] > dc_upper[i] or close[i] > pivot_6h[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Donchian(20) breakout on 6h with volume surge and weekly pivot direction filter.
# Weekly pivot from prior week provides institutional reference point.
# Long when price breaks above Donchian upper with volume surge and above weekly pivot.
# Short when price breaks below Donchian lower with volume surge and below weekly pivot.
# Weekly pivot acts as dynamic support/resistance, reducing false breakouts.
# Works in bull markets (breakouts above pivot) and bear markets (breakdowns below pivot).
# Volume surge confirms institutional participation. Cooldown reduces trade frequency.
# Target: 60-120 total trades over 4 years (15-30/year) to avoid fee drag.
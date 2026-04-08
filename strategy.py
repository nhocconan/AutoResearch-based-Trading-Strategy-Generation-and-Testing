#!/usr/bin/env python3
# [24931] 6h_1d_donchian_weekly_pivot_v1
# Hypothesis: 6-hour Donchian(20) breakout with 1-day pivot direction filter and volume confirmation.
# Long when price breaks above 20-period high with volume > 1.8x average and price > 1-day pivot point.
# Short when price breaks below 20-period low with volume > 1.8x average and price < 1-day pivot point.
# Exit when price reverts to 10-period moving average or volume drops below 1.3x average.
# Uses weekly pivot from 1-day data for trend bias, effective in both trending and ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_donchian_weekly_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for weekly pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot from daily data (using previous week's data)
    # Weekly pivot = (Prior week high + prior week low + prior week close) / 3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Need at least 5 days for a week
    weekly_pivot = np.full(len(close_1d), np.nan)
    for i in range(4, len(close_1d)):
        # Use previous 5 days (prior week)
        week_high = np.max(high_1d[i-4:i+1])
        week_low = np.min(low_1d[i-4:i+1])
        week_close = close_1d[i]
        weekly_pivot[i] = (week_high + week_low + week_close) / 3.0
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 10-period moving average for exit
    ma_10 = np.full(n, np.nan)
    for i in range(10, n):
        ma_10[i] = np.mean(close[i-10:i])
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align weekly pivot to 6-hour timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ma_10[i]) or np.isnan(vol_ma[i]) or np.isnan(weekly_pivot_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        
        if position == 1:  # Long
            # Exit: price returns to 10-period MA or volume drops below 1.3x average
            if price <= ma_10[i] or vol_ratio < 1.3:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price returns to 10-period MA or volume drops below 1.3x average
            if price >= ma_10[i] or vol_ratio < 1.3:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high with volume expansion and above weekly pivot
            if price > donchian_high[i] and vol_ratio > 1.8 and price > weekly_pivot_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low with volume expansion and below weekly pivot
            elif price < donchian_low[i] and vol_ratio > 1.8 and price < weekly_pivot_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
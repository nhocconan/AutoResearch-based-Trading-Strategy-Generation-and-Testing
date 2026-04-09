#!/usr/bin/env python3
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
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot levels (using prior week's OHLC)
    weekly_high = np.full(len(df_d), np.nan)
    weekly_low = np.full(len(df_d), np.nan)
    weekly_close = np.full(len(df_d), np.nan)
    weekly_pivot = np.full(len(df_d), np.nan)
    weekly_r1 = np.full(len(df_d), np.nan)
    weekly_s1 = np.full(len(df_d), np.nan)
    
    # Calculate weekly aggregation from daily data
    for i in range(len(df_d)):
        if i >= 7:  # Need at least 7 days for weekly
            week_high = np.max(df_d['high'].iloc[i-7:i])
            week_low = np.min(df_d['low'].iloc[i-7:i])
            week_close = df_d['close'].iloc[i-1]
            weekly_high[i] = week_high
            weekly_low[i] = week_low
            weekly_close[i] = week_close
            weekly_pivot[i] = (week_high + week_low + week_close) / 3.0
            weekly_r1[i] = 2 * weekly_pivot[i] - week_low
            weekly_s1[i] = 2 * weekly_pivot[i] - week_high
    
    # Calculate Donchian channel (20-period) on 6h data
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(n):
        if i >= 20:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_d, weekly_s1)
    
    # Volume confirmation: 20-period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly pivot
            if close[i] < weekly_pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly pivot
            if close[i] > weekly_pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high with volume confirmation
            if (close[i] > donchian_high[i] and 
                volume[i] > vol_ma_20[i] * 1.5):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low with volume confirmation
            elif (close[i] < donchian_low[i] and 
                  volume[i] > vol_ma_20[i] * 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction filter and volume confirmation
# Uses weekly pivot levels (calculated from 1d data aggregated to weekly) to determine structural bias
# Only takes breakouts in the direction of the weekly pivot (above weekly pivot = long bias, below = short bias)
# Volume confirmation ensures breakouts have conviction
# Position size 0.25 to manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag
# Weekly pivot provides structural context that works in both bull and bear markets by adapting to higher timeframe bias

name = "6h_1d_weekly_pivot_donchian_volume_v1"
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
    
    # Load 1d data ONCE before loop for weekly pivot calculation and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot from 1d data (using prior week's data to avoid look-ahead)
    weekly_pivot = np.full(len(df_1d), np.nan)
    weekly_r1 = np.full(len(df_1d), np.nan)
    weekly_s1 = np.full(len(df_1d), np.nan)
    
    # Group 1d data into weeks (starting from Monday)
    # We'll use a simple approach: every 7 days is a week, using prior week's data
    for i in range(len(df_1d)):
        if i < 7:
            weekly_pivot[i] = np.nan
            weekly_r1[i] = np.nan
            weekly_s1[i] = np.nan
        else:
            # Use prior week's OHLC (7 days ago to 1 day ago) to calculate current week's pivot
            week_start = max(0, i - 7)
            week_end = i - 1  # prior week (excluding current day to avoid look-ahead)
            
            if week_end >= week_start and week_end < len(df_1d):
                week_high = np.max(df_1d['high'].iloc[week_start:week_end+1].values)
                week_low = np.min(df_1d['low'].iloc[week_start:week_end+1].values)
                week_close = df_1d['close'].iloc[week_end]
                
                pivot = (week_high + week_low + week_close) / 3.0
                weekly_pivot[i] = pivot
                weekly_r1[i] = 2 * pivot - week_low
                weekly_s1[i] = 2 * pivot - week_high
            else:
                weekly_pivot[i] = np.nan
                weekly_r1[i] = np.nan
                weekly_s1[i] = np.nan
    
    # Calculate 1d ATR(14) for dynamic position sizing (optional filter)
    tr_1d = np.full(len(df_1d), np.nan)
    for i in range(1, len(df_1d)):
        tr = max(
            df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
            abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
            abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
        )
        tr_1d[i] = tr
    
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i < 14:
            atr_1d[i] = np.nan
        elif i == 14:
            atr_1d[i] = np.nanmean(tr_1d[1:15])
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Align 1d weekly pivot levels to 6h timeframe
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_6h = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_6h = align_htf_to_ltf(prices, df_1d, weekly_s1)
    atr_1d_6h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 20-period Donchian channels on 6h
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_6h[i]) or 
            np.isnan(weekly_r1_6h[i]) or 
            np.isnan(weekly_s1_6h[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume[i] > 1.3 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit conditions: price closes below Donchian low OR weekly pivot turns bearish (price < weekly pivot)
            if close[i] < donchian_low[i] or close[i] < weekly_pivot_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price closes above Donchian high OR weekly pivot turns bullish (price > weekly pivot)
            if close[i] > donchian_high[i] or close[i] > weekly_pivot_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Donchian breakout in direction of weekly pivot bias with volume confirmation
            if volume_confirm:
                # Long breakout: price closes above Donchian high AND price > weekly pivot (bullish bias)
                if close[i] > donchian_high[i] and close[i] > weekly_pivot_6h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below Donchian low AND price < weekly pivot (bearish bias)
                elif close[i] < donchian_low[i] and close[i] < weekly_pivot_6h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
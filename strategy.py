#!/usr/bin/env python3
# 6h_1d_weekly_pivot_breakout_v1
# Hypothesis: 6-hour breakout of weekly pivot levels with daily trend filter and volume confirmation.
# Weekly pivot levels (calculated from prior week OHLC) act as strong support/resistance.
# Long when price breaks above weekly R1 with price > daily EMA50 and volume > 2.0x 20-bar average.
# Short when price breaks below weekly S1 with price < daily EMA50 and volume > 2.0x 20-bar average.
# Exit when price returns to opposite weekly pivot level (S1 for longs, R1 for shorts).
# Works in bull markets via breakout continuation and in bear markets via mean reversion at extreme levels.
# Target: 50-150 total trades over 4 years (12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_weekly_pivot_breakout_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema = np.mean(close_1d[:50])  # Initialize with first 50-period average
        multiplier = 2 / (50 + 1)
        ema_50_1d[49] = ema
        for i in range(50, len(close_1d)):
            ema = (close_1d[i] - ema) * multiplier + ema
            ema_50_1d[i] = ema
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate weekly pivot levels from prior week OHLC
    # Need to resample 1d to weekly using actual weekly boundaries
    # We'll compute weekly OHLC from daily data
    weekly_high = []
    weekly_low = []
    weekly_close = []
    
    # Group daily data into weeks (assuming data starts on Monday)
    # Simple approach: every 7 days
    for i in range(0, len(df_1d), 7):
        if i + 6 < len(df_1d):
            week_high = np.max(df_1d['high'].iloc[i:i+7])
            week_low = np.min(df_1d['low'].iloc[i:i+7])
            week_close = df_1d['close'].iloc[i+6]
            weekly_high.append(week_high)
            weekly_low.append(week_low)
            weekly_close.append(week_close)
        else:
            # Handle incomplete week at end
            week_high = np.max(df_1d['high'].iloc[i:])
            week_low = np.min(df_1d['low'].iloc[i:])
            week_close = df_1d['close'].iloc[-1]
            weekly_high.append(week_high)
            weekly_low.append(week_low)
            weekly_close.append(week_close)
    
    # Calculate pivot points for each week
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    weekly_pivot = []
    weekly_r1 = []
    weekly_s1 = []
    
    for i in range(len(weekly_high)):
        h = weekly_high[i]
        l = weekly_low[i]
        c = weekly_close[i]
        p = (h + l + c) / 3
        r1 = 2 * p - l
        s1 = 2 * p - h
        weekly_pivot.append(p)
        weekly_r1.append(r1)
        weekly_s1.append(s1)
    
    # Now we need to map weekly values to daily timeframe
    # Each weekly value applies to the 7 days that followed
    daily_pivot = np.full(len(df_1d), np.nan)
    daily_r1 = np.full(len(df_1d), np.nan)
    daily_s1 = np.full(len(df_1d), np.nan)
    
    for week_idx in range(len(weekly_pivot)):
        start_day = week_idx * 7
        end_day = min(start_day + 7, len(df_1d))
        if start_day < len(df_1d):
            daily_pivot[start_day:end_day] = weekly_pivot[week_idx]
            daily_r1[start_day:end_day] = weekly_r1[week_idx]
            daily_s1[start_day:end_day] = weekly_s1[week_idx]
    
    # Align weekly pivot levels to 6h timeframe
    # Note: We use the prior week's levels (so we shift by 1 week)
    # This ensures we're using completed weekly data only
    daily_pivot_shifted = np.roll(daily_pivot, 7)
    daily_r1_shifted = np.roll(daily_r1, 7)
    daily_s1_shifted = np.roll(daily_s1, 7)
    
    # Set first week to NaN (no prior week data)
    daily_pivot_shifted[:7] = np.nan
    daily_r1_shifted[:7] = np.nan
    daily_s1_shifted[:7] = np.nan
    
    pivot_aligned = align_htf_to_ltf(prices, df_1d, daily_pivot_shifted)
    r1_aligned = align_htf_to_ltf(prices, df_1d, daily_r1_shifted)
    s1_aligned = align_htf_to_ltf(prices, df_1d, daily_s1_shifted)
    
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
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or below S1 level
            if close[i] <= s1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above R1 level
            if close[i] >= r1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above R1 with trend and volume filters
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > vol_ma_20[i] * 2.0):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below S1 with trend and volume filters
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > vol_ma_20[i] * 2.0):
                position = -1
                signals[i] = -0.25
    
    return signals
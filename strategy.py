#!/usr/bin/env python3
"""
6H Weekly Pivot Breakout with Volume Confirmation and Daily Trend Filter
Long when price breaks above weekly R3 with volume expansion AND daily EMA trend up
Short when price breaks below weekly S3 with volume expansion AND daily EMA trend down
Exit when price crosses back to weekly pivot point
Uses weekly pivots from actual weekly data (not resampled) for structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_breakout_volume_daily_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly Pivot Points (from actual weekly data) ===
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Calculate pivot points for each weekly bar
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Standard pivot: P = (H + L + C) / 3
    # R3 = P + 2*(H - L)
    # S3 = P - 2*(H - L)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_range = weekly_high - weekly_low
    weekly_r3 = weekly_pivot + 2 * weekly_range
    weekly_s3 = weekly_pivot - 2 * weekly_range
    
    # Align to 6h timeframe (with shift(1) for completed weekly bars only)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s3)
    
    # === Daily trend filter (EMA 21) ===
    df_daily = get_htf_data(prices, '1d')
    ema_daily = pd.Series(df_daily['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)  # Avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(ema_daily_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses back below weekly pivot
            if close[i] < pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses back above weekly pivot
            if close[i] > pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need expanding volume (above average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: Weekly pivot breakout with volume confirmation AND daily trend filter
            if close[i] > r3_aligned[i] and ema_daily_aligned[i] > ema_daily_aligned[i-1]:
                # Breakout above weekly R3 with rising daily EMA -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < s3_aligned[i] and ema_daily_aligned[i] < ema_daily_aligned[i-1]:
                # Breakdown below weekly S3 with falling daily EMA -> short
                position = -1
                signals[i] = -0.25
    
    return signals
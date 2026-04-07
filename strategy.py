#!/usr/bin/env python3
"""
6H Weekly Pivot Breakout with Volume Confirmation and Daily Trend Filter
Long when price breaks above weekly S3 pivot with expanding volume AND daily EMA trend up
Short when price breaks below weekly R3 pivot with expanding volume AND daily EMA trend down
Exit when price crosses back to weekly pivot point
Uses weekly pivot levels from prior week and daily EMA for trend filter.
Target: 50-150 total trades over 4 years = 12-37/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_breakout_volume_1d_trend_v1"
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
    
    # === Weekly Pivot Points (from prior week) ===
    df_weekly = get_htf_data(prices, '1w')
    # Use prior week's data to avoid look-ahead: shift(1) built into align_htf_to_ltf
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate pivot points for each week
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    r2 = pp + (weekly_high - weekly_low)
    s2 = pp - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pp - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pp)
    r4 = pp + 3 * (weekly_high - weekly_low)
    s4 = pp - 3 * (weekly_high - weekly_low)
    
    # Align weekly pivots to 6h timeframe (with shift(1) for prior week only)
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)  # Avoid division by zero
    
    # === Daily trend filter (EMA 21) ===
    df_daily = get_htf_data(prices, '1d')
    ema_daily = pd.Series(df_daily['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(ema_daily_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses back below weekly pivot point
            if close[i] < pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses back above weekly pivot point
            if close[i] > pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need expanding volume (above average)
            if vol_ratio[i] < 1.3:
                signals[i] = 0.0
                continue
            
            # Entry: Weekly pivot breakout with volume confirmation AND daily trend filter
            if close[i] > s3_aligned[i] and ema_daily_aligned[i] > ema_daily_aligned[i-1]:
                # Break below S3 with rising daily EMA -> long (mean reversion from extreme)
                position = 1
                signals[i] = 0.25
            elif close[i] < r3_aligned[i] and ema_daily_aligned[i] < ema_daily_aligned[i-1]:
                # Break above R3 with falling daily EMA -> short (mean reversion from extreme)
                position = -1
                signals[i] = -0.25
    
    return signals
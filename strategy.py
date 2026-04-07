#!/usr/bin/env python3
"""
6H Weekly Pivot Breakout with Volume Confirmation and Daily Trend Filter
Strategy: Break above R3 or below S3 weekly pivot levels with expanding volume.
Long when price breaks above R3 and daily EMA 50 is rising.
Short when price breaks below S3 and daily EMA 50 is falling.
Exit when price crosses back to weekly pivot point.
Weekly pivots act as strong support/resistance; breakouts with volume indicate institutional interest.
Daily EMA filter ensures alignment with intermediate trend to avoid counter-trend trades.
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
    
    # === Weekly pivot points (using weekly OHLC) ===
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot: PP = (H + L + C) / 3
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    r2 = pp + (weekly_high - weekly_low)
    s2 = pp - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pp - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pp)
    
    # Align weekly pivots to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === Daily trend filter (EMA 50) ===
    df_daily = get_htf_data(prices, '1d')
    ema_daily = pd.Series(df_daily['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
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
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: Weekly pivot breakout with volume confirmation AND daily trend filter
            if close[i] > r3_aligned[i] and ema_daily_aligned[i] > ema_daily_aligned[i-1]:
                # Breakout above R3 with rising daily EMA -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < s3_aligned[i] and ema_daily_aligned[i] < ema_daily_aligned[i-1]:
                # Breakdown below S3 with falling daily EMA -> short
                position = -1
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
"""
1d_1w_Camarilla_Breakout_Trend_v1
Hypothesis: Trade breakouts of daily R3/S3 only when weekly price is within S3/R3 (range-bound context) and ADX(14) > 25 on daily. Use tight position sizing (0.25) to limit drawdown. Target: 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_Breakout_Trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === DAILY DATA ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot calculation
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels (daily)
    r3_1d = close_1d + range_1d * 1.1
    s3_1d = close_1d - range_1d * 1.1
    
    # === WEEKLY DATA ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot calculation
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Weekly Camarilla levels
    r3_1w = close_1w + range_1w * 1.1
    s3_1w = close_1w - range_1w * 1.1
    
    # === DAILY ADX(14) FOR TREND STRENGTH ===
    if len(df_1d) >= 14:
        # True Range
        tr1 = np.abs(high_1d[1:] - low_1d[1:])
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with index
        
        # Directional Movement
        dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                           np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
        dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                            np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
        dm_plus = np.concatenate([[np.nan], dm_plus])
        dm_minus = np.concatenate([[np.nan], dm_minus])
        
        # Smoothed values
        def smooth_wilder(arr, period):
            result = np.full_like(arr, np.nan)
            if len(arr) < period:
                return result
            # First value: simple average
            result[period-1] = np.nanmean(arr[1:period])
            # Subsequent values: Wilder smoothing
            for i in range(period, len(arr)):
                if not np.isnan(result[i-1]):
                    result[i] = (result[i-1] * (period-1) + arr[i]) / period
            return result
        
        atr = smooth_wilder(tr, 14)
        dm_plus_smooth = smooth_wilder(dm_plus, 14)
        dm_minus_smooth = smooth_wilder(dm_minus, 14)
        
        # DI+ and DI-
        di_plus = np.where(atr != 0, dm_plus_smooth / atr * 100, 0)
        di_minus = np.where(atr != 0, dm_minus_smooth / atr * 100, 0)
        
        # DX and ADX
        dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
        adx = smooth_wilder(dx, 14)
    else:
        adx = np.full(len(df_1d), np.nan)
    
    # Align to daily timeframe (no change needed as we're already on daily)
    r3_1d_aligned = r3_1d
    s3_1d_aligned = s3_1d
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    adx_aligned = adx  # already on daily
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # start after warmup for ADX
        # Skip if indicators not available
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Weekly range-bound context: price between S3 and R3
        weekly_range = (close[i] > s3_1w_aligned[i]) & (close[i] < r3_1w_aligned[i])
        
        # Breakout entries at S3/R3 with trend and weekly range filters
        long_setup = (close[i] > r3_1d_aligned[i]) and strong_trend and weekly_range
        short_setup = (close[i] < s3_1d_aligned[i]) and strong_trend and weekly_range
        
        # Exit when price returns to daily pivot (mean reversion)
        pivot_1d_aligned = pivot_1d  # already on daily
        exit_long = close[i] < pivot_1d_aligned[i]
        exit_short = close[i] > pivot_1d_aligned[i]
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
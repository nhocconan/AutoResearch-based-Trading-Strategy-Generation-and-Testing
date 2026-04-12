#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Precompute hour filter for 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly pivot
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Get 1d data for daily pivot and trend context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's OHLC)
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    # Classic pivot point formula
    pivot_1w = (prev_week_high + prev_week_low + prev_week_close) / 3
    r1_1w = 2 * pivot_1w - prev_week_low
    s1_1w = 2 * pivot_1w - prev_week_high
    r2_1w = pivot_1w + (prev_week_high - prev_week_low)
    s2_1w = pivot_1w - (prev_week_high - prev_week_low)
    
    # Calculate daily pivot points
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    prev_day_close = df_1d['close'].shift(1).values
    
    pivot_1d = (prev_day_high + prev_day_low + prev_day_close) / 3
    r1_1d = 2 * pivot_1d - prev_day_low
    s1_1d = 2 * pivot_1d - prev_day_high
    r2_1d = pivot_1d + (prev_day_high - prev_day_low)
    s2_1d = pivot_1d - (prev_day_high - prev_day_low)
    r3_1d = pivot_1d + 2 * (prev_day_high - prev_day_low)
    s3_1d = pivot_1d - 2 * (prev_day_high - prev_day_low)
    r4_1d = pivot_1d + 3 * (prev_day_high - prev_day_low)
    s4_1d = pivot_1d - 3 * (prev_day_high - prev_day_low)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Align daily pivot levels to 6h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate 1d ADX for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = minus_dm[0] = np.nan
    
    # Smooth TR and DM
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[1:period]) if not np.isnan(arr[1:period]).all() else np.nan
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]) and not np.isnan(arr[i]):
                result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr_1d = smooth_wilder(tr, 14)
    plus_di_1d = 100 * smooth_wilder(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * smooth_wilder(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = smooth_wilder(dx_1d, 14)
    
    # Align ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(r2_1w_aligned[i]) or 
            np.isnan(s2_1w_aligned[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(pivot_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or 
            np.isnan(s4_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 20 for trending market
        trending = adx_1d_aligned[i] > 20
        
        # Weekly bias: price above/below weekly pivot
        weekly_bullish = close[i] > pivot_1w_aligned[i]
        weekly_bearish = close[i] < pivot_1w_aligned[i]
        
        # Entry conditions: break of daily R4/S4 with weekly bias + trend
        long_entry = (close[i] > r4_1d_aligned[i]) and weekly_bullish and trending
        short_entry = (close[i] < s4_1d_aligned[i]) and weekly_bearish and trending
        
        # Exit conditions: return to daily pivot or trend weakening
        long_exit = (close[i] < pivot_1d_aligned[i]) or (adx_1d_aligned[i] < 15)
        short_exit = (close[i] > pivot_1d_aligned[i]) or (adx_1d_aligned[i] < 15)
        
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
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_1d_pivot_adx_breakout_v1"
timeframe = "6h"
leverage = 1.0
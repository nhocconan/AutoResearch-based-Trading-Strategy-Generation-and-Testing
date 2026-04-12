#!/usr/bin/env python3
"""
4h_12h_Camarilla_Pivot_Trend_Filter_v1
Hypothesis: Use 12h Camarilla pivot levels for mean reversion in range markets, filtered by 1d ADX trend strength. 
Long when price touches S3 with ADX < 25 (range), short when touches R3 with ADX < 25. Exit at pivot.
Avoids trend days (ADX >= 25) to prevent losses in strong moves. Works in bull via mean reversion in ranges, 
in bear via same logic as ADX filter adapts to regime. Target: 20-50 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Camarilla_Pivot_Trend_Filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 12h data for Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Previous 12h bar for pivot calculation
    prev_high = df_12h['high'].iloc[-2]
    prev_low = df_12h['low'].iloc[-2]
    prev_close = df_12h['close'].iloc[-2]
    
    # Calculate Camarilla levels
    range_val = prev_high - prev_low
    if range_val <= 0:
        return np.zeros(n)
    
    camarilla_mult = 1.1 / 6  # ~0.1833
    camarilla_s3 = prev_close - range_val * camarilla_mult * 4  # S3 = Close - 4 * 1.1/6 * Range
    camarilla_r3 = prev_close + range_val * camarilla_mult * 4  # R3 = Close + 4 * 1.1/6 * Range
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3
    
    # Align Camarilla levels to 4h
    camarilla_s3_array = np.full(len(df_12h), camarilla_s3)
    camarilla_r3_array = np.full(len(df_12h), camarilla_r3)
    camarilla_pivot_array = np.full(len(df_12h), camarilla_pivot)
    
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3_array)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3_array)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_12h, camarilla_pivot_array)
    
    # 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX
        return np.zeros(n)
    
    # Calculate ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    def smooth_wmma(arr, period):
        """Wilder's smoothing (EMA with alpha=1/period)"""
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(arr[1:period])  # Skip first NaN
        # Wilder smoothing: alpha = 1/period
        alpha = 1.0 / period
        for i in range(period, len(arr)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
        return result
    
    tr_smooth = smooth_wmma(tr, 14)
    dm_plus_smooth = smooth_wmma(dm_plus, 14)
    dm_minus_smooth = smooth_wmma(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smooth_wmma(dx, 14)
    
    # Align ADX to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Range filter: only trade when ADX < 25 (no strong trend)
        range_market = adx_aligned[i] < 25
        
        # Mean reversion signals at Camarilla S3/R3
        long_signal = range_market and close[i] <= camarilla_s3_aligned[i]
        short_signal = range_market and close[i] >= camarilla_r3_aligned[i]
        
        # Exit at pivot
        long_exit = close[i] >= camarilla_pivot_aligned[i]
        short_exit = close[i] <= camarilla_pivot_aligned[i]
        
        # Signal logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
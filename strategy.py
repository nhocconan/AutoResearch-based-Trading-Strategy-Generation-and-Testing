# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
6h_ADX_WilliamsAlligator_1dTrend
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) trend detection combined with ADX filter on 1d timeframe.
In trending markets, Alligator lines align in order (Lips > Teeth > Jaw for uptrend, reverse for downtrend).
ADX > 25 confirms strong trend, filtering whipsaws. Works in both bull (trend continuation) and bear (trend reversals).
Target: 50-150 total trades over 4 years (12-37/year).
"""

name = "6h_ADX_WilliamsAlligator_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 1d Williams Alligator (13,8,5 smoothed with SMMA)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # SMMA (Smoothed Moving Average) function
    def smma(arr, period):
        res = np.full_like(arr, np.nan)
        if len(arr) >= period:
            res[period-1] = np.mean(arr[:period])
            for i in range(period, len(arr)):
                res[i] = (res[i-1] * (period-1) + arr[i]) / period
        return res
    
    # Alligator lines: Jaw(13,8), Teeth(8,5), Lips(5,3)
    jaw_raw = smma((high_1d + low_1d) / 2, 13)
    teeth_raw = smma((high_1d + low_1d) / 2, 8)
    lips_raw = smma((high_1d + low_1d) / 2, 5)
    
    # Smooth further with SMMA
    jaw = smma(jaw_raw, 8)
    teeth = smma(teeth_raw, 5)
    lips = smma(lips_raw, 3)
    
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # 1d ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
        minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
        plus_dm = np.concatenate([[0], plus_dm])
        minus_dm = np.concatenate([[0], minus_dm])
        
        # Smoothed TR, +DM, -DM
        def smooth_wilder(arr, period):
            res = np.full_like(arr, np.nan)
            if len(arr) >= period:
                res[period-1] = np.nansum(arr[1:period+1])
                for i in range(period, len(arr)):
                    res[i] = res[i-1] - (res[i-1] / period) + arr[i]
            return res
        
        tr_smooth = smooth_wilder(tr, period)
        plus_dm_smooth = smooth_wilder(plus_dm, period)
        minus_dm_smooth = smooth_wilder(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # DX and ADX
        dx = np.where(tr_smooth != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = np.full_like(dx, np.nan)
        if len(dx) >= period:
            adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])
            for i in range(2*period-1, len(dx)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 30)  # warmup for Alligator and ADX
    
    for i in range(start_idx, n):
        if np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or np.isnan(adx_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend detection: Alligator alignment
        # Uptrend: Lips > Teeth > Jaw
        # Downtrend: Jaw > Teeth > Lips
        is_uptrend = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        is_downtrend = jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i]
        
        # ADX filter: strong trend
        strong_trend = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long: Uptrend + strong ADX
            if is_uptrend and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: Downtrend + strong ADX
            elif is_downtrend and strong_trend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Trend weakens or reverses
            if not (is_uptrend and strong_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Trend weakens or reverses
            if not (is_downtrend and strong_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
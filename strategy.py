#!/usr/bin/env python3
"""
6h ADX + Williams Alligator System
Trend strength + smoothed moving average crossover system
Works in both bull/bear markets by filtering for strong trends
"""

name = "6h_ADX_Alligator_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(arr, period):
    """Smoothed Moving Average"""
    n = len(arr)
    result = np.full(n, np.nan)
    if n < period:
        return result
    sma = np.mean(arr[:period])
    result[period-1] = sma
    for i in range(period, n):
        result[i] = (result[i-1] * (period-1) + arr[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for ADX and Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX on 1d
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM
    period_adx = 14
    tr_sum = np.full_like(tr, np.nan)
    plus_dm_sum = np.full_like(plus_dm, np.nan)
    minus_dm_sum = np.full_like(minus_dm, np.nan)
    
    if len(tr) >= period_adx:
        tr_sum[period_adx-1] = np.nansum(tr[1:period_adx+1])
        plus_dm_sum[period_adx-1] = np.nansum(plus_dm[1:period_adx+1])
        minus_dm_sum[period_adx-1] = np.nansum(minus_dm[1:period_adx+1])
        
        for i in range(period_adx, len(tr)):
            tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / period_adx) + tr[i]
            plus_dm_sum[i] = plus_dm_sum[i-1] - (plus_dm_sum[i-1] / period_adx) + plus_dm[i]
            minus_dm_sum[i] = minus_dm_sum[i-1] - (minus_dm_sum[i-1] / period_adx) + minus_dm[i]
    
    # DI+ and DI-
    plus_di = np.full_like(tr, np.nan)
    minus_di = np.full_like(tr, np.nan)
    valid_tr = tr_sum != 0
    plus_di[valid_tr] = 100 * plus_dm_sum[valid_tr] / tr_sum[valid_tr]
    minus_di[valid_tr] = 100 * minus_dm_sum[valid_tr] / tr_sum[valid_tr]
    
    # DX and ADX
    dx = np.full_like(tr, np.nan)
    di_sum = plus_di + minus_di
    valid_di = di_sum != 0
    dx[valid_di] = 100 * np.abs(plus_di[valid_di] - minus_di[valid_di]) / di_sum[valid_di]
    
    adx = np.full_like(tr, np.nan)
    if len(dx) >= period_adx:
        adx[period_adx-1] = np.nanmean(dx[period_adx-1:2*period_adx-1])
        for i in range(2*period_adx-1, len(dx)):
            adx[i] = (adx[i-1] * (period_adx-1) + dx[i]) / period_adx
    
    # Align ADX to 6m
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Williams Alligator on 1d (SMA-based smoothed)
    jaw = smma(close_1d, 13)  # Blue line
    teeth = smma(close_1d, 8)  # Red line
    lips = smma(close_1d, 5)   # Green line
    
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(period_adx*2-1, 13)  # Need ADX and Alligator ready
    
    for i in range(start_idx, n):
        if np.isnan(adx_aligned[i]) or np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # ADX trend strength filter (>25 = strong trend)
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + strong trend
            if lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: Jaws > Teeth > Lips (bearish alignment) + strong trend
            elif jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i] and strong_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator alignment breaks or trend weakens
            if not (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]) or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator alignment breaks or trend weakens
            if not (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]) or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
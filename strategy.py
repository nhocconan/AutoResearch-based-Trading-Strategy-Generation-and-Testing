#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_DailyTrend_VolumeBreakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels (primary HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard floor trader pivots)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Resistance and Support levels (standard pivot)
    R1 = pivot_1w + (range_1w * 1.0) / 2
    S1 = pivot_1w - (range_1w * 1.0) / 2
    R2 = pivot_1w + range_1w
    S2 = pivot_1w - range_1w
    R3 = pivot_1w + range_1w * 2.0
    S3 = pivot_1w - range_1w * 2.0
    
    # Align weekly pivot levels to 6h timeframe
    R1_1w_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_1w_aligned = align_htf_to_ltf(prices, df_1w, S1)
    R2_1w_aligned = align_htf_to_ltf(prices, df_1w, R2)
    S2_1w_aligned = align_htf_to_ltf(prices, df_1w, S2)
    R3_1w_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_1w_aligned = align_htf_to_ltf(prices, df_1w, S3)
    
    # Daily EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume spike detection (6h timeframe)
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(R1_1w_aligned[i]) or 
            np.isnan(S1_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma20[i]  # Require strong volume spike
        
        if position == 0:
            # Long: Price breaks above weekly R2 with daily uptrend and volume spike
            if close[i] > R2_1w_aligned[i] and close[i] > ema_1d_aligned[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S2 with daily downtrend and volume spike
            elif close[i] < S2_1w_aligned[i] and close[i] < ema_1d_aligned[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below weekly S1 or trend turns down
            if close[i] < S1_1w_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above weekly R1 or trend turns up
            if close[i] > R1_1w_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
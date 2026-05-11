#!/usr/bin/env python3
name = "6h_WeeklyPivot_Bias_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w Weekly Pivot levels (using previous week)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot calculation
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r1_1w = pivot_1w + (range_1w * 0.382)  # Weekly R1
    s1_1w = pivot_1w - (range_1w * 0.382)  # Weekly S1
    
    # Align weekly levels to 6h timeframe
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (6h volume > 2.0x 20-period average)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 2.0 * volume_ma20
    
    # Session filter: 08-20 UTC (most liquid session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close above weekly R1, above 1d EMA50, volume confirmation, session
            if close[i] > r1_1w_aligned[i] and close[i] > ema_50_1d_aligned[i] and volume_filter[i] and session_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below weekly S1, below 1d EMA50, volume confirmation, session
            elif close[i] < s1_1w_aligned[i] and close[i] < ema_50_1d_aligned[i] and volume_filter[i] and session_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close below weekly S1 or below 1d EMA50
            if close[i] < s1_1w_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close above weekly R1 or above 1d EMA50
            if close[i] > r1_1w_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
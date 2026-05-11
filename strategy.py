#!/usr/bin/env python3
name = "6h_WeeklyPivot_BullBear_Switch_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot (using previous week)
    pivot = (high_1w[:-1] + low_1w[:-1] + close_1w[:-1]) / 3
    r1 = 2 * pivot - low_1w[:-1]
    s1 = 2 * pivot - high_1w[:-1]
    
    # Align to 6h (previous week's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, np.concatenate([[np.nan], pivot]))
    r1_aligned = align_htf_to_ltf(prices, df_1w, np.concatenate([[np.nan], r1]))
    s1_aligned = align_htf_to_ltf(prices, df_1w, np.concatenate([[np.nan], s1]))
    
    # 1d trend filter (EMA 50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter (24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 24)
    
    for i in range(start_idx, n):
        if np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > weekly R1 + above 1d EMA + volume
            if close[i] > r1_aligned[i] and close[i] > ema_1d_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < weekly S1 + below 1d EMA + volume
            elif close[i] < s1_aligned[i] and close[i] < ema_1d_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < weekly S1 or below 1d EMA
            if close[i] < s1_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > weekly R1 or above 1d EMA
            if close[i] > r1_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
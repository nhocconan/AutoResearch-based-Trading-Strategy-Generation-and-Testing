#!/usr/bin/env python3
# 6H_WeeklyPivot_Donchian_Breakout_Trend_Filter
# Hypothesis: Combine weekly pivot levels with daily Donchian breakouts on 6h timeframe.
# Long when: price breaks above 6h Donchian(20) high AND above weekly pivot point AND 1d trend up.
# Short when: price breaks below 6h Donchian(20) low AND below weekly pivot point AND 1d trend down.
# Uses weekly pivot for institutional reference and Donchian for breakout confirmation.
# Works in bull/bear by following 1d trend direction, reducing whipsaws.

name = "6H_WeeklyPivot_Donchian_Breakout_Trend_Filter"
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
    volume = prices['volume'].values
    
    # 1d Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1w data for weekly pivot
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly pivot from previous week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point: (H + L + C) / 3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    
    # 1d trend filter: EMA 34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly pivot to 6h (wait for weekly close)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    # Align 1d EMA to 6h (wait for daily close)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history
    
    for i in range(start_idx, n):
        if np.isnan(weekly_pivot_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        is_uptrend = close[i] > ema_34_1d_aligned[i]
        is_downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long entry: Donchian breakout + above weekly pivot + uptrend
            if (high[i] > high_20[i] and 
                close[i] > weekly_pivot_aligned[i] and 
                is_uptrend and
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Donchian breakdown + below weekly pivot + downtrend
            elif (low[i] < low_20[i] and 
                  close[i] < weekly_pivot_aligned[i] and 
                  is_downtrend and
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low or below weekly pivot
            if low[i] < low_20[i] or close[i] < weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high or above weekly pivot
            if high[i] > high_20[i] or close[i] > weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
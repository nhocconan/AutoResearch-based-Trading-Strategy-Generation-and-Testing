#!/usr/bin/env python3
name = "6h_WeeklyPivot_Breakout_DailyTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Weekly data for pivot points (Pivot, R1, S1)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Pivot Point = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    # Weekly R1 = 2 * Pivot - Low
    r1_1w = 2 * pivot_1w - low_1w
    # Weekly S1 = 2 * Pivot - High
    s1_1w = 2 * pivot_1w - high_1w
    
    # Align weekly levels to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 200  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R1, uptrend (price > EMA50), volume filter
            if (close[i] > r1_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1, downtrend (price < EMA50), volume filter
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below pivot (mean reversion to weekly mean)
            if close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above pivot (mean reversion to weekly mean)
            if close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
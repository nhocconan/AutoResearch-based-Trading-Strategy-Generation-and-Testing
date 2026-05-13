#!/usr/bin/env python3
name = "12h_PivotReversal_1dTrend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1D data ONCE for pivot points and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1D EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate previous day's pivot points (R1, S1)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    pivot_1d = (high_1d[:-1] + low_1d[:-1] + close_1d[:-1]) / 3.0
    r1_1d = 2 * pivot_1d - low_1d[:-1]
    s1_1d = 2 * pivot_1d - high_1d[:-1]
    
    # Prepend NaN for first day (no previous day data)
    pivot_1d = np.concatenate([[np.nan], pivot_1d])
    r1_1d = np.concatenate([[np.nan], r1_1d])
    s1_1d = np.concatenate([[np.nan], s1_1d])
    
    # Align 1D indicators to 12H timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(pivot_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA50
        price_above_ema = close[i] > ema50_1d_aligned[i]
        price_below_ema = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # LONG: Price touches S1 support in uptrend
            if price_above_ema and low[i] <= s1_1d_aligned[i] * 1.002:  # Allow 0.2% slippage
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches R1 resistance in downtrend
            elif price_below_ema and high[i] >= r1_1d_aligned[i] * 0.998:  # Allow 0.2% slippage
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches pivot or trend changes
            if high[i] >= pivot_1d_aligned[i] * 0.998 or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches pivot or trend changes
            if low[i] <= pivot_1d_aligned[i] * 1.002 or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
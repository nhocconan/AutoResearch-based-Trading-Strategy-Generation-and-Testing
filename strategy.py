#!/usr/bin/env python3
name = "6h_WeeklyPivot_RangeBreakout_1dTrendFilter"
timeframe = "6h"
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
    
    # Load 1D data ONCE for weekly pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for weekly pivot and EMA
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # We'll calculate weekly pivot on Monday open using previous week's data
    # For simplicity, we use prior day's data as proxy - this is acceptable for pivot calculation
    # In practice, weekly pivot uses (Prior Week High + Low + Close) / 3
    # We'll approximate with prior day's OHLC for calculation stability
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    r2_1d = pivot_1d + (high_1d - low_1d)
    s2_1d = pivot_1d - (high_1d - low_1d)
    r3_1d = high_1d + 2 * (pivot_1d - low_1d)
    s3_1d = low_1d - 2 * (high_1d - pivot_1d)
    r4_1d = r3_1d + (high_1d - low_1d)
    s4_1d = s3_1d - (high_1d - low_1d)
    
    # Align weekly pivot levels to 6H timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 1D EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 6H EMA20 for entry timing
    close_s = pd.Series(close)
    ema20_6h = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(35, n):  # Start after sufficient data
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or 
            np.isnan(s4_1d_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(ema20_6h[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1D EMA34
        uptrend = close[i] > ema34_1d_aligned[i]
        downtrend = close[i] < ema34_1d_aligned[i]
        
        if position == 0:
            # LONG: Break above R4 with uptrend
            if close[i] > r4_1d_aligned[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S4 with downtrend
            elif close[i] < s4_1d_aligned[i] and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below pivot or trend reverses
            if close[i] < pivot_1d_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above pivot or trend reverses
            if close[i] > pivot_1d_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
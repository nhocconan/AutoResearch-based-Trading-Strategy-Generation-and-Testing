#!/usr/bin/env python3
name = "6h_WeeklyPivot_Trend_Continuation"
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
    
    # Load weekly data ONCE for pivot and trend
    df_wk = get_htf_data(prices, '1w')
    if len(df_wk) < 10:
        return np.zeros(n)
    
    high_wk = df_wk['high'].values
    low_wk = df_wk['low'].values
    close_wk = df_wk['close'].values
    
    # Weekly pivot points: P = (H+L+C)/3
    pivot_wk = (high_wk + low_wk + close_wk) / 3.0
    r1_wk = 2 * pivot_wk - low_wk
    s1_wk = 2 * pivot_wk - high_wk
    
    # Align weekly pivot to 6H timeframe
    pivot_wk_aligned = align_htf_to_ltf(prices, df_wk, pivot_wk)
    r1_wk_aligned = align_htf_to_ltf(prices, df_wk, r1_wk)
    s1_wk_aligned = align_htf_to_ltf(prices, df_wk, s1_wk)
    
    # 6H EMA20 for trend filter
    close_s = pd.Series(close)
    ema20_6h = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: current volume > 20-period average
    volume_s = pd.Series(volume)
    vol_ma20 = volume_s.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(pivot_wk_aligned[i]) or np.isnan(r1_wk_aligned[i]) or 
            np.isnan(s1_wk_aligned[i]) or np.isnan(ema20_6h[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to EMA20
        price_above_ema = close[i] > ema20_6h[i]
        price_below_ema = close[i] < ema20_6h[i]
        
        if position == 0:
            # LONG: Break above R1 with volume and uptrend
            if (close[i] > r1_wk_aligned[i]) and price_above_ema and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S1 with volume and downtrend
            elif (close[i] < s1_wk_aligned[i]) and price_below_ema and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below weekly pivot
            if close[i] < pivot_wk_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above weekly pivot
            if close[i] > pivot_wk_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
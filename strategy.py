#!/usr/bin/env python3
name = "6h_SR_Flip_WeeklyTrend_HTF_Confirm"
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
    
    # === Weekly Trend (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    # Weekly EMA40 trend filter (slow enough to capture major trend)
    ema40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    # === Dynamic Support/Resistance from Daily Pivots (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    # Prior day's high/low/close for pivot calculation
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    pivot = (prev_high + prev_low + prev_close) / 3.0
    # Support 1 and Resistance 1 (core S/R levels)
    s1 = (2 * pivot) - prev_high
    r1 = (2 * pivot) - prev_low
    # Align to 6t
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # === Volume Filter (LTF) ===
    # 20-period volume moving average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema40_1w_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # LONG: Price > R1, above weekly EMA40, volume confirmation
            if close[i] > r1_aligned[i] and close[i] > ema40_1w_aligned[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: Price < S1, below weekly EMA40, volume confirmation
            elif close[i] < s1_aligned[i] and close[i] < ema40_1w_aligned[i] and vol_ok:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S1 (support flip) or weekly trend turns down
            if close[i] < s1_aligned[i] or close[i] < ema40_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above R1 (resistance flip) or weekly trend turns up
            if close[i] > r1_aligned[i] or close[i] > ema40_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
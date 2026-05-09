#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_WeeklyPivot_Breakout_Trend_Filter_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly pivot points (using previous week's data)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly pivots to 12h
    pivot_1w_12h = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_12h = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_12h = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_12h = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_12h = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Align daily EMA50 to 12h
    ema50_1d_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_1w_12h[i]) or np.isnan(r1_1w_12h[i]) or np.isnan(s1_1w_12h[i]) or
            np.isnan(r2_1w_12h[i]) or np.isnan(s2_1w_12h[i]) or np.isnan(ema50_1d_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pivot = pivot_1w_12h[i]
        r1 = r1_1w_12h[i]
        s1 = s1_1w_12h[i]
        r2 = r2_1w_12h[i]
        s2 = s2_1w_12h[i]
        trend = ema50_1d_12h[i]
        
        if position == 0:
            # Long: break above R1 with trend alignment
            if close[i] > r1 and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with trend alignment
            elif close[i] < s1 and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below pivot or trend reversal
            if close[i] < pivot or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above pivot or trend reversal
            if close[i] > pivot or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
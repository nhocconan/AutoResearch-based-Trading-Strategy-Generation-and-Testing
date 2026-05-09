#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_Breakout_DailyTrend_Volume"
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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Get daily data for trend and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Daily EMA34 for trend
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Daily volume average
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly pivots to 6h (with extra delay for pivot confirmation)
    pivot_1w_6h = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_6h = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_6h = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_6h = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_6h = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Align daily indicators to 6h
    ema34_1d_6h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    vol_avg_1d_6h = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 40
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_1w_6h[i]) or np.isnan(r1_1w_6h[i]) or np.isnan(s1_1w_6h[i]) or
            np.isnan(r2_1w_6h[i]) or np.isnan(s2_1w_6h[i]) or np.isnan(ema34_1d_6h[i]) or
            np.isnan(vol_avg_1d_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pivot = pivot_1w_6h[i]
        r1 = r1_1w_6h[i]
        s1 = s1_1w_6h[i]
        r2 = r2_1w_6h[i]
        s2 = s2_1w_6h[i]
        trend = ema34_1d_6h[i]
        vol_avg = vol_avg_1d_6h[i]
        vol_ok = volume[i] > vol_avg * 1.8
        
        if position == 0:
            # Long: break above R2 with volume and above daily trend
            if close[i] > r2 and vol_ok and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Short: break below S2 with volume and below daily trend
            elif close[i] < s2 and vol_ok and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below S1 or trend reversal
            if close[i] < s1 or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above R1 or trend reversal
            if close[i] > r1 or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
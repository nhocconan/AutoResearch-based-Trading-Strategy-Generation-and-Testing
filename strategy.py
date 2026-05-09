#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivotBreakout_VolumeTrend"
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
    
    # Get weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot levels (H, L, C from previous week)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point and support/resistance levels
    weekly_pivot = np.zeros(len(close_1w))
    weekly_r1 = np.zeros(len(close_1w))
    weekly_s1 = np.zeros(len(close_1w))
    weekly_r2 = np.zeros(len(close_1w))
    weekly_s2 = np.zeros(len(close_1w))
    
    for i in range(1, len(close_1w)):
        H = high_1w[i-1]  # Previous week high
        L = low_1w[i-1]   # Previous week low
        C = close_1w[i-1] # Previous week close
        
        P = (H + L + C) / 3.0
        weekly_pivot[i] = P
        weekly_r1[i] = 2*P - L
        weekly_s1[i] = 2*P - H
        weekly_r2[i] = P + (H - L)
        weekly_s2[i] = P - (H - L)
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    
    # Get daily trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period MA
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(weekly_r2_aligned[i]) or 
            np.isnan(weekly_s2_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Long: Break above weekly R2 with volume and above daily EMA trend
            if close[i] > weekly_r2_aligned[i] and vol_ok and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below weekly S2 with volume and below daily EMA trend
            elif close[i] < weekly_s2_aligned[i] and vol_ok and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below weekly pivot or trend reversal
            if close[i] < weekly_pivot_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above weekly pivot or trend reversal
            if close[i] > weekly_pivot_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
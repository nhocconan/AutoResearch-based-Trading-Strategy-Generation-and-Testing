#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation (resistance/support)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot: (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Resistance 1: 2*P - L
    r1_1w = 2 * pivot_1w - low_1w
    # Support 1: 2*P - H
    s1_1w = 2 * pivot_1w - high_1w
    # Resistance 2: P + (H - L)
    r2_1w = pivot_1w + (high_1w - low_1w)
    # Support 2: P - (H - L)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Align weekly pivots to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Get daily data for trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d volume average (20-period)
    vol_1d = pd.Series(df_1d['volume'].values)
    vol_ma20_1d = vol_1d.rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Current volume for confirmation
    vol_series = pd.Series(volume)
    vol_ma20_current = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(r2_1w_aligned[i]) or 
            np.isnan(s2_1w_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma20_1d_aligned[i]) or np.isnan(vol_ma20_current[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20_current[i]
        
        if position == 0:
            # Long: Break above weekly R2 with volume and above 1d EMA trend
            if close[i] > r2_1w_aligned[i] and vol_ok and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below weekly S2 with volume and below 1d EMA trend
            elif close[i] < s2_1w_aligned[i] and vol_ok and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below weekly R1 or trend reversal
            if close[i] < r1_1w_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above weekly S1 or trend reversal
            if close[i] > s1_1w_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
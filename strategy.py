#!/usr/bin/env python3
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
    
    # Get daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly high/low from last 5 trading days (weekly period)
    high_5d = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    low_5d = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    close_prev = pd.Series(close_1d).shift(1).values  # Previous day close
    
    # Standard weekly pivot point formula (using weekly high/low/prev close)
    pivot = (high_5d + low_5d + close_prev) / 3.0
    r1 = 2 * pivot - low_5d
    s1 = 2 * pivot - high_5d
    r2 = pivot + (high_5d - low_5d)
    s2 = pivot - (high_5d - low_5d)
    
    # Align weekly pivots to daily timeframe (since we're using 1d timeframe)
    pivot_1d = align_htf_to_ltf(prices, df_1d, pivot)
    r1_1d = align_htf_to_ltf(prices, df_1d, r1)
    s1_1d = align_htf_to_ltf(prices, df_1d, s1)
    r2_1d = align_htf_to_ltf(prices, df_1d, r2)
    s2_1d = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume confirmation (20-period MA on daily)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma20[i]) or 
            np.isnan(pivot_1d[i]) or 
            np.isnan(r1_1d[i]) or 
            np.isnan(s1_1d[i]) or 
            np.isnan(r2_1d[i]) or 
            np.isnan(s2_1d[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period average
        volume_filter = volume[i] > (1.3 * volume_ma20[i])
        
        if position == 0:
            # Long: break above R2 with volume
            if close[i] > r2_1d[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below S2 with volume
            elif close[i] < s2_1d[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below R1 or volume dries up
            if close[i] < r1_1d[i] or volume[i] < volume_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above S1 or volume dries up
            if close[i] > s1_1d[i] or volume[i] < volume_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyPivot_R2_S2_Breakout_Volume"
timeframe = "1d"
leverage = 1.0
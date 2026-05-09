#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_WeeklyPivot_Breakout"
timeframe = "12h"
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
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    pivot_w = (high_w + low_w + close_w) / 3.0
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    r2_w = pivot_w + (high_w - low_w)
    s2_w = pivot_w - (high_w - low_w)
    
    # Align weekly pivot levels to 12h timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_w, pivot_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_w, s1_w)
    r2_w_aligned = align_htf_to_ltf(prices, df_w, r2_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_w, s2_w)
    
    # Volume filter: current volume > 1.2 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(pivot_w_aligned[i]) or 
            np.isnan(r1_w_aligned[i]) or
            np.isnan(s1_w_aligned[i]) or
            np.isnan(r2_w_aligned[i]) or
            np.isnan(s2_w_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pivot_val = pivot_w_aligned[i]
        r1_val = r1_w_aligned[i]
        s1_val = s1_w_aligned[i]
        r2_val = r2_w_aligned[i]
        s2_val = s2_w_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: Price above R1 + volume filter
            if close[i] > r1_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: Price below S1 + volume filter
            elif close[i] < s1_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below pivot
            if close[i] < pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above pivot
            if close[i] > pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
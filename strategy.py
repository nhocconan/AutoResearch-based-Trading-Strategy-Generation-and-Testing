#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyPivot_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 10:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA for trend
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate weekly pivot points
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Volume filter: current daily volume > 1.5 * 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 20)  # EMA20 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema20_1w_aligned[i]) or
            np.isnan(pivot_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema20_val = ema20_1w_aligned[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: close above R1 + above weekly EMA trend + volume filter
            if close[i] > r1_val and close[i] > ema20_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: close below S1 + below weekly EMA trend + volume filter
            elif close[i] < s1_val and close[i] < ema20_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below weekly EMA trend
            if close[i] < ema20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above weekly EMA trend
            if close[i] > ema20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
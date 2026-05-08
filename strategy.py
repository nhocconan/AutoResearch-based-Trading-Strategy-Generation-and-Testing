#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points (HTF)
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 30:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2P-L, S1 = 2P-H
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    pivot_w = (high_w + low_w + close_w) / 3.0
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    
    # Align weekly pivot levels to 6h timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_w, pivot_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_w, s1_w)
    
    # Get daily data for trend filter
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 30:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend
    close_d = df_d['close'].values
    ema34_d = pd.Series(close_d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily EMA34 to 6h timeframe
    ema34_d_aligned = align_htf_to_ltf(prices, df_d, ema34_d)
    
    # Calculate 6-day average volume for volume confirmation
    vol_ma6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_w_aligned[i]) or np.isnan(r1_w_aligned[i]) or 
            np.isnan(s1_w_aligned[i]) or np.isnan(ema34_d_aligned[i]) or
            np.isnan(vol_ma6[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current volume above 6-period average
        vol_filter = volume[i] > vol_ma6[i]
        
        if position == 0:
            # Long: price above weekly R1 + daily EMA34 uptrend + volume
            if (close[i] > r1_w_aligned[i] and 
                close[i] > ema34_d_aligned[i] and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly S1 + daily EMA34 downtrend + volume
            elif (close[i] < s1_w_aligned[i] and 
                  close[i] < ema34_d_aligned[i] and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price below weekly pivot OR daily EMA34 turns down
            if (close[i] < pivot_w_aligned[i] or 
                close[i] < ema34_d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price above weekly pivot OR daily EMA34 turns up
            if (close[i] > pivot_w_aligned[i] or 
                close[i] > ema34_d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
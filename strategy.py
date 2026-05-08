#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_Breakout_TrendFilter_v1"
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
    
    # Weekly data for pivot and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Weekly pivot points from previous week
    P = np.zeros(n)  # Pivot
    R1 = np.zeros(n)  # Resistance 1
    S1 = np.zeros(n)  # Support 1
    R2 = np.zeros(n)  # Resistance 2
    S2 = np.zeros(n)  # Support 2
    
    for i in range(1, len(close_1w)):
        high_prev = high_1w[i-1]
        low_prev = low_1w[i-1]
        close_prev = close_1w[i-1]
        pivot = (high_prev + low_prev + close_prev) / 3.0
        
        P[i] = pivot
        R1[i] = 2 * pivot - low_prev
        S1[i] = 2 * pivot - high_prev
        R2[i] = pivot + (high_prev - low_prev)
        S2[i] = pivot - (high_prev - low_prev)
    
    # Align weekly pivot levels to 6h timeframe
    P_aligned = align_htf_to_ltf(prices, df_1w, P)
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    R2_aligned = align_htf_to_ltf(prices, df_1w, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1w, S2)
    
    # Volume filter: current volume > 1.3x 50-period average (6h timeframe)
    vol_ma50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (1.3 * vol_ma50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(P_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1, weekly uptrend, volume filter
            long_cond = (close[i] > R1_aligned[i] and 
                        ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1] and
                        volume_filter[i])
            
            # Short: price breaks below S1, weekly downtrend, volume filter
            short_cond = (close[i] < S1_aligned[i] and 
                         ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1] and
                         volume_filter[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below pivot P
            if close[i] < P_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above pivot P
            if close[i] > P_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
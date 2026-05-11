#!/usr/bin/env python3
name = "6h_ElderRay_Power_Trend"
timeframe = "6h"
leverage = 1.0

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
    
    # 1d data for Elder Ray and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 13-period EMA for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power_1d = high_1d - ema13_1d
    # Bear Power = Low - EMA13
    bear_power_1d = low_1d - ema13_1d
    
    # Smooth the power values with 6-period EMA
    bull_power_smooth_1d = pd.Series(bull_power_1d).ewm(span=6, adjust=False, min_periods=6).mean().values
    bear_power_smooth_1d = pd.Series(bear_power_1d).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    # Align to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_smooth_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_smooth_1d)
    
    # 6-day EMA for trend filter (1d EMA6)
    ema6_1d = pd.Series(close_1d).ewm(span=6, adjust=False, min_periods=6).mean().values
    ema6_1d_aligned = align_htf_to_ltf(prices, df_1d, ema6_1d)
    
    # 6h EMA13 for entry timing
    ema13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema6_1d_aligned[i]) or np.isnan(ema13_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull power rising and above zero, price above EMA13, uptrend (EMA6 rising)
            if (bull_power_aligned[i] > 0 and 
                bull_power_aligned[i] > bull_power_aligned[i-1] and
                close[i] > ema13_6h[i] and
                ema6_1d_aligned[i] > ema6_1d_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: Bear power falling and below zero, price below EMA13, downtrend (EMA6 falling)
            elif (bear_power_aligned[i] < 0 and 
                  bear_power_aligned[i] < bear_power_aligned[i-1] and
                  close[i] < ema13_6h[i] and
                  ema6_1d_aligned[i] < ema6_1d_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bull power turns negative or price crosses below EMA13
            if (bull_power_aligned[i] <= 0 or 
                close[i] < ema13_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bear power turns positive or price crosses above EMA13
            if (bear_power_aligned[i] >= 0 or 
                close[i] > ema13_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
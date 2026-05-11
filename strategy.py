#!/usr/bin/env python3
name = "1d_Weekly_Camarilla_R1_S1_Breakout_Trend"
timeframe = "1d"
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
    
    # Get weekly data for Camarilla and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly Camarilla levels (based on previous week)
    high_prev = df_1w['high'].shift(1).values
    low_prev = df_1w['low'].shift(1).values
    close_prev = df_1w['close'].shift(1).values
    range_prev = high_prev - low_prev
    
    # R1 and S1 levels
    r1 = close_prev + 1.1 * range_prev / 12
    s1 = close_prev - 1.1 * range_prev / 12
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up = close_1w > ema50
    
    # Align weekly indicators to daily
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    trend_up_aligned = align_htf_to_ltf(prices, df_1w, trend_up)
    
    # Volume moving average (20-period) for confirmation
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(trend_up_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above R1 + uptrend + volume confirmation
            if (close[i] > r1_aligned[i] and 
                trend_up_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.30
                position = 1
            # Short: price crosses below S1 + downtrend + volume confirmation
            elif (close[i] < s1_aligned[i] and 
                  not trend_up_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price crosses below S1 or trend changes
            if (close[i] < s1_aligned[i] or not trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price crosses above R1 or trend changes
            if (close[i] > r1_aligned[i] or trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
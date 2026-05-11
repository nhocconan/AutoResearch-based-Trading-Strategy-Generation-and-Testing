#!/usr/bin/env python3
name = "6h_Camarilla_R3_S3_Breakout_1wTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily and weekly data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly trend: EMA34 > EMA89 = uptrend, EMA34 < EMA89 = downtrend
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_1w = pd.Series(df_1w['close']).ewm(span=89, adjust=False, min_periods=89).mean().values
    weekly_uptrend = ema34_1w > ema89_1w
    weekly_downtrend = ema34_1w < ema89_1w
    
    # Align weekly trend
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend)
    
    # Previous day's close for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels (using previous day's range)
    R4 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 2
    R3 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 4
    S4 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(R4_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(S4_aligned[i]) or 
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 in weekly uptrend with volume surge
            if (close[i] > R3_aligned[i] and 
                weekly_uptrend_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 in weekly downtrend with volume surge
            elif (close[i] < S3_aligned[i] and 
                  weekly_downtrend_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below R1 or weekly trend changes
            # R1 calculation: prev_close + 1.1*(prev_high-prev_low)*1.1/6
            R1 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 6
            R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
            if (close[i] < R1_aligned[i] or not weekly_uptrend_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above S1 or weekly trend changes
            # S1 calculation: prev_close - 1.1*(prev_high-prev_low)*1.1/6
            S1 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 6
            S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
            if (close[i] > S1_aligned[i] or not weekly_downtrend_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
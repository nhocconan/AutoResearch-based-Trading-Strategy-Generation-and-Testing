#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_12hTrend_Filter
Hypothesis: Use Camarilla pivot levels R3/S3 on daily timeframe for breakout signals, filtered by 12h EMA50 trend direction. Enter long when price breaks above R3 in 12h uptrend, short when price breaks below S3 in 12h downtrend. Exit on opposite break or trend reversal. Designed for low-frequency, high-conviction trades with strong trend alignment.
"""

name = "4h_Camarilla_R3S3_Breakout_12hTrend_Filter"
timeframe = "4h"
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
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Avoid NaN in first row
    high_prev[0] = high_prev[1] if len(high_prev) > 1 else high_prev[0]
    low_prev[0] = low_prev[1] if len(low_prev) > 1 else low_prev[0]
    close_prev[0] = close_prev[1] if len(close_prev) > 1 else close_prev[0]
    
    # Camarilla calculations
    rang = high_prev - low_prev
    R3 = close_prev + rang * 1.1 / 4
    S3 = close_prev - rang * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need previous day data
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(close_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 12h trend determination
        trend_12h_up = close_12h_aligned[i] > ema_50_12h_aligned[i]
        trend_12h_down = close_12h_aligned[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 in 12h uptrend
            if close[i] > R3_aligned[i] and trend_12h_up:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 in 12h downtrend
            elif close[i] < S3_aligned[i] and trend_12h_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 or trend turns down
            if close[i] < S3_aligned[i] or not trend_12h_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 or trend turns up
            if close[i] > R3_aligned[i] or not trend_12h_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
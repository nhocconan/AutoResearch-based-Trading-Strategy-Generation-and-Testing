#!/usr/bin/env python3
"""
1H_Camarilla_4hTrend_1dSMA50
Hypothesis: Use 4h Camarilla levels (R3/S3) for entry timing in 1h timeframe, with 1d SMA50 as trend filter.
In bull market: price above daily SMA50, look for long entries when 1h closes above 4h R3.
In bear market: price below daily SMA50, look for short entries when 1h closes below 4h S3.
This reduces overtrading by requiring alignment between 1h timing and higher timeframe structure.
Target: 15-30 trades/year per symbol (60-120 total over 4 years).
"""
name = "1H_Camarilla_4hTrend_1dSMA50"
timeframe = "1h"
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
    
    # Get 4h data for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (R3, S3)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla: R3 = close + (high - low) * 1.1 / 4, S3 = close - (high - low) * 1.1 / 4
    r3 = close_4h + (high_4h - low_4h) * 1.1 / 4
    s3 = close_4h - (high_4h - low_4h) * 1.1 / 4
    
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    
    # Get 1d data for SMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily SMA50
    close_1d = df_1d['close'].values
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup for SMA50
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(sma_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above daily SMA50, 1h close above 4h R3, and in session
            if (close[i] > sma_50_1d_aligned[i] and 
                close[i] > r3_aligned[i] and 
                session_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: price below daily SMA50, 1h close below 4h S3, and in session
            elif (close[i] < sma_50_1d_aligned[i] and 
                  close[i] < s3_aligned[i] and 
                  session_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below daily SMA50
            if close[i] < sma_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above daily SMA50
            if close[i] > sma_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
#!/usr/bin/env python3
"""
1h_4h1d_Triple_SMA_Trend
Hypothesis: Use 4h for primary trend (SMA50) and 1d for secondary trend (SMA200), with 1h for entry timing via SMA20 pullback in trending direction. 
Designed for low trade frequency (target: 15-37/year) by requiring alignment of multiple timeframes and avoiding entries in choppy markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4h SMA50 for primary trend
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    sma50_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        for i in range(49, len(close_4h)):
            sma50_4h[i] = np.mean(close_4h[i-49:i+1])
    sma50_4h_aligned = align_htf_to_ltf(prices, df_4h, sma50_4h)
    
    # 1d SMA200 for secondary trend
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    sma200_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        for i in range(199, len(close_1d)):
            sma200_1d[i] = np.mean(close_1d[i-199:i+1])
    sma200_1d_aligned = align_htf_to_ltf(prices, df_1d, sma200_1d)
    
    # 1h SMA20 for entry timing
    sma20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            sma20[i] = np.mean(close[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    start_idx = max(199, 50, 20)
    
    for i in range(start_idx, n):
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if (np.isnan(sma50_4h_aligned[i]) or np.isnan(sma200_1d_aligned[i]) or 
            np.isnan(sma20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above both SMAs and pulling back to SMA20
            if (close[i] > sma50_4h_aligned[i] and close[i] > sma200_1d_aligned[i] and 
                close[i] <= sma20[i] * 1.01):  # Allow small penetration above SMA20
                signals[i] = 0.20
                position = 1
            # Short: price below both SMAs and pulling back to SMA20
            elif (close[i] < sma50_4h_aligned[i] and close[i] < sma200_1d_aligned[i] and 
                  close[i] >= sma20[i] * 0.99):  # Allow small penetration below SMA20
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price closes below SMA20 or trend breaks
            if (close[i] < sma20[i] * 0.99 or close[i] < sma50_4h_aligned[i] or 
                close[i] < sma200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price closes above SMA20 or trend breaks
            if (close[i] > sma20[i] * 1.01 or close[i] > sma50_4h_aligned[i] or 
                close[i] > sma200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h1d_Triple_SMA_Trend"
timeframe = "1h"
leverage = 1.0
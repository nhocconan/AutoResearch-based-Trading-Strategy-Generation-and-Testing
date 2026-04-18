#!/usr/bin/env python3
"""
12h_KAMA_Trend_Direction_Plus_Volume_Spike
Hypothesis: KAMA adapts to market noise, providing reliable trend direction on 12h timeframe. 
Combined with volume spike confirmation on 12h, this filters false breakouts. 
Designed for low trade frequency (12-37/year) with strong performance in both bull and bear markets by avoiding whipsaws in ranging conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for KAMA calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily data
    # Parameters: ER period=10, Fast EMA=2, Slow EMA=30
    kama_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 30:  # Need enough data for slow EMA
        # Efficiency Ratio
        change = np.abs(np.diff(close_1d, n=10))  # 10-period change
        volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=1)  # 10-period volatility
        # Pad the beginning with NaN
        change = np.concatenate([np.full(10, np.nan), change])
        volatility = np.concatenate([np.full(10, np.nan), volatility])
        er = np.where(volatility != 0, change / volatility, 0)
        # Smoothing constants
        sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
        # Initialize KAMA
        kama_1d[29] = np.mean(close_1d[0:30])  # Start with simple average
        for i in range(30, len(close_1d)):
            if not np.isnan(sc[i]):
                kama_1d[i] = kama_1d[i-1] + sc[i] * (close_1d[i] - kama_1d[i-1])
            else:
                kama_1d[i] = kama_1d[i-1]
    
    # Align 1-day KAMA to 12h timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Volume spike: current 12h volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA with volume spike
            if close[i] > kama_1d_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA with volume spike
            elif close[i] < kama_1d_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below KAMA
            if close[i] < kama_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above KAMA
            if close[i] > kama_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Trend_Direction_Plus_Volume_Spike"
timeframe = "12h"
leverage = 1.0
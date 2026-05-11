#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_Volume
Hypothesis: Combines Camarilla pivot levels (R1/S1 from 1d) with 12h EMA trend filter and volume confirmation.
In bull markets, price breaks above R1 with upward trend; in bear markets, price breaks below S1 with downward trend.
Volume confirmation filters false breakouts. Target: 20-50 trades per year on 4h timeframe.
"""

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume"
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
    
    # === 1D Data for Camarilla Pivots ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    R1 = np.zeros(len(close_1d))
    S1 = np.zeros(len(close_1d))
    for i in range(len(close_1d)):
        if i == 0:
            R1[i] = close_1d[i]
            S1[i] = close_1d[i]
        else:
            R1[i] = close_1d[i-1] + (high_1d[i-1] - low_1d[i-1]) * 1.1 / 12
            S1[i] = close_1d[i-1] - (high_1d[i-1] - low_1d[i-1]) * 1.1 / 12
    
    # Align Camarilla levels to 4h
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # === 12H Data for Trend Filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h EMA50 for trend
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # === Volume Filter ===
    # 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with upward trend and volume confirmation
            if (close[i] > R1_aligned[i] and 
                ema50_12h_aligned[i] > close[i] and  # Uptrend: price above EMA
                volume[i] > vol_ma[i] * 1.5):       # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with downward trend and volume confirmation
            elif (close[i] < S1_aligned[i] and 
                  ema50_12h_aligned[i] < close[i] and  # Downtrend: price below EMA
                  volume[i] > vol_ma[i] * 1.5):        # Volume spike
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 or trend turns down
            if (close[i] < S1_aligned[i] or 
                ema50_12h_aligned[i] < close[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price breaks above R1 or trend turns up
            if (close[i] > R1_aligned[i] or 
                ema50_12h_aligned[i] > close[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals
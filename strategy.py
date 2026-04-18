#!/usr/bin/env python3
"""
4h_MACD_Convergence_With_1D_Volume_Filter
Hypothesis: On 4h timeframe, use MACD convergence (MACD line approaching signal line) combined with volume expansion to identify trend continuation. Enter long when MACD line crosses above signal line and volume > 1.5x average, short when MACD line crosses below signal line and volume > 1.5x average. Use 1D EMA(50) filter to avoid counter-trend trades in bear markets. Targets 20-30 trades/year by requiring MACD cross + volume filter + EMA filter, with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate MACD components
    fast = 12
    slow = 26
    signal_period = 9
    
    # EMA calculations
    ema_fast = np.full(n, np.nan)
    ema_slow = np.full(n, np.nan)
    
    # Calculate EMA using Wilder's smoothing (alpha = 2/(period+1))
    alpha_fast = 2.0 / (fast + 1)
    alpha_slow = 2.0 / (slow + 1)
    
    # Initialize EMAs
    ema_fast[fast-1] = np.mean(close[0:fast])
    ema_slow[slow-1] = np.mean(close[0:slow])
    
    for i in range(fast, n):
        ema_fast[i] = alpha_fast * close[i] + (1 - alpha_fast) * ema_fast[i-1]
    
    for i in range(slow, n):
        ema_slow[i] = alpha_slow * close[i] + (1 - alpha_slow) * ema_slow[i-1]
    
    macd_line = ema_fast - ema_slow
    
    # Signal line EMA of MACD
    signal_line = np.full(n, np.nan)
    alpha_signal = 2.0 / (signal_period + 1)
    
    # Find first valid MACD value for signal line initialization
    first_valid = np.where(~np.isnan(macd_line))[0]
    if len(first_valid) > 0:
        idx = first_valid[0]
        signal_line[idx] = macd_line[idx]
        for i in range(idx + 1, n):
            signal_line[i] = alpha_signal * macd_line[i] + (1 - alpha_signal) * signal_line[i-1]
    
    # MACD histogram
    macd_hist = macd_line - signal_line
    
    # Volume moving average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # Get 1D data for EMA filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1D
    ema50_1d = np.full(len(close_1d), np.nan)
    alpha_1d = 2.0 / (50 + 1)
    ema50_1d[49] = np.mean(close_1d[0:50])
    for i in range(50, len(close_1d)):
        ema50_1d[i] = alpha_1d * close_1d[i] + (1 - alpha_1d) * ema50_1d[i-1]
    
    # Align 1D EMA to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, slow, vol_period)  # need all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(macd_line[i]) or np.isnan(signal_line[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: MACD line crosses above signal line AND volume expansion AND price above 1D EMA50
            if (macd_line[i-1] <= signal_line[i-1] and macd_line[i] > signal_line[i] and 
                volume[i] > 1.5 * vol_ma[i] and close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: MACD line crosses below signal line AND volume expansion AND price below 1D EMA50
            elif (macd_line[i-1] >= signal_line[i-1] and macd_line[i] < signal_line[i] and 
                  volume[i] > 1.5 * vol_ma[i] and close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: MACD line crosses below signal line
            if macd_line[i] < signal_line[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: MACD line crosses above signal line
            if macd_line[i] > signal_line[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_MACD_Convergence_With_1D_Volume_Filter"
timeframe = "4h"
leverage = 1.0
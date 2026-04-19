#!/usr/bin/env python3

# 4h_MACD_Zero_Cross_With_Volume_Confirmation
# Hypothesis: MACD line crossing zero on 4h timeframe indicates trend changes. 
# MACD zero cross provides clear entry/exit signals with low latency. 
# Volume confirmation filters out false signals. Works in both bull and bear markets
# by capturing momentum shifts. Target 20-50 trades/year for low friction.

name = "4h_MACD_Zero_Cross_With_Volume_Confirmation"
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
    
    # MACD calculation
    def calculate_macd(close_prices, fast=12, slow=26, signal=9):
        # Calculate EMAs
        ema_fast = pd.Series(close_prices).ewm(span=fast, adjust=False, min_periods=fast).mean().values
        ema_slow = pd.Series(close_prices).ewm(span=slow, adjust=False, min_periods=slow).mean().values
        macd_line = ema_fast - ema_slow
        signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False, min_periods=signal).mean().values
        return macd_line, signal_line
    
    # Get 4h data for MACD calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate MACD on 4h data
    macd_line, signal_line = calculate_macd(df_4h['close'].values)
    macd_line_aligned = align_htf_to_ltf(prices, df_4h, macd_line)
    signal_line_aligned = align_htf_to_ltf(prices, df_4h, signal_line)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(macd_line_aligned[i]) or np.isnan(signal_line_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: MACD crosses above zero with volume confirmation
            if (macd_line_aligned[i] > 0 and macd_line_aligned[i-1] <= 0 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: MACD crosses below zero with volume confirmation
            elif (macd_line_aligned[i] < 0 and macd_line_aligned[i-1] >= 0 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if MACD crosses below zero
            if macd_line_aligned[i] < 0 and macd_line_aligned[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if MACD crosses above zero
            if macd_line_aligned[i] > 0 and macd_line_aligned[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
name = "6h_Fibonacci_23p6_Retracement_1dTrend_Volume"
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
    
    # 1d data for trend filter and Fibonacci levels
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA(50) on 1d for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 6-month high/low (180 days) for Fibonacci retracement
    # Using 180 trading days = ~6 months
    high_180 = pd.Series(high_1d).rolling(window=180, min_periods=180).max().values
    low_180 = pd.Series(low_1d).rolling(window=180, min_periods=180).min().values
    
    # 23.6% Fibonacci retracement level: Low + (High - Low) * 0.236
    fib_23p6 = low_180 + (high_180 - low_180) * 0.236
    
    # Align 1d data to 6h
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    fib_23p6_aligned = align_htf_to_ltf(prices, df_1d, fib_23p6)
    high_180_aligned = align_htf_to_ltf(prices, df_1d, high_180)
    low_180_aligned = align_htf_to_ltf(prices, df_1d, low_180)
    
    # Volume spike: current volume > 2.0x 50-period average
    vol_avg = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 180  # need enough data for 180-day lookback
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(fib_23p6_aligned[i]) or 
            np.isnan(high_180_aligned[i]) or np.isnan(low_180_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price pulls back to 23.6% Fib + 1d trend up + volume spike
            if (close[i] <= fib_23p6_aligned[i] * 1.01 and  # within 1% of fib level
                close[i] >= fib_23p6_aligned[i] * 0.99 and
                close[i] > ema50_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price rallies to 23.6% Fib + 1d trend down + volume spike
            elif (close[i] <= fib_23p6_aligned[i] * 1.01 and 
                  close[i] >= fib_23p6_aligned[i] * 0.99 and
                  close[i] < ema50_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks above 61.8% Fib or below 23.6% with trend change
            fib_61p8 = low_180_aligned[i] + (high_180_aligned[i] - low_180_aligned[i]) * 0.618
            if close[i] > fib_61p8 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks below 38.2% Fib or above 23.6% with trend change
            fib_38p2 = low_180_aligned[i] + (high_180_aligned[i] - low_180_aligned[i]) * 0.382
            if close[i] < fib_38p2 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
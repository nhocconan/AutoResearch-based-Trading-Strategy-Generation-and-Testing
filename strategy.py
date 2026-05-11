#!/usr/bin/env python3
"""
4h_Fibonacci_Retracement_Trend_v1
Hypothesis: Price often retraces to key Fibonacci levels (38.2%, 50%, 61.8%) of the prior daily swing before resuming the trend.
In bull markets, buy the dip at 0.618 retracement of the prior day's upmove; in bear markets, sell the rally at 0.382 retracement of the prior day's downmove.
Uses 1d swing high/low for Fibonacci levels and 4h EMA50 for trend filter. Target: 50-150 trades over 4 years (12-37/year) on 4h timeframe.
"""

name = "4h_Fibonacci_Retracement_Trend_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 1D Data for Swing Points ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate prior day's swing (for Fibonacci levels)
    swing_high = high_1d  # current day's high
    swing_low = low_1d    # current day's low
    swing_range = swing_high - swing_low
    
    # Fibonacci levels: 0.382, 0.5, 0.618
    fib_382 = swing_low + 0.382 * swing_range
    fib_500 = swing_low + 0.500 * swing_range
    fib_618 = swing_low + 0.618 * swing_range
    
    # Align Fibonacci levels to 4h
    fib_382_aligned = align_htf_to_ltf(prices, df_1d, fib_382)
    fib_500_aligned = align_htf_to_ltf(prices, df_1d, fib_500)
    fib_618_aligned = align_htf_to_ltf(prices, df_1d, fib_618)
    
    # === 4H Data for Trend Filter ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # 4h EMA50 for trend
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(fib_382_aligned[i]) or 
            np.isnan(fib_500_aligned[i]) or 
            np.isnan(fib_618_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price at 0.618 fib support in uptrend (buy the dip)
            if close[i] <= fib_618_aligned[i] * 1.001 and ema50_4h_aligned[i] < close[i]:
                signals[i] = 0.25
                position = 1
            # Short: price at 0.382 fib resistance in downtrend (sell the rally)
            elif close[i] >= fib_382_aligned[i] * 0.999 and ema50_4h_aligned[i] > close[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below 0.382 fib or trend turns down
            if close[i] < fib_382_aligned[i] * 0.999 or ema50_4h_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price breaks above 0.618 fib or trend turns up
            if close[i] > fib_618_aligned[i] * 1.001 or ema50_4h_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals
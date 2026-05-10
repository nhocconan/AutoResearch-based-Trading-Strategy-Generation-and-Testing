#/usr/bin/env python3
"""
6h_MarketStructure_1dTrend_Filter
Hypothesis: Buy when price makes a higher low above prior swing low during uptrend (1d EMA50 up),
sell when price makes a lower high below prior swing high during downtrend (1d EMA50 down).
This captures trend continuation after pullbacks, working in both bull (buy dips) and bear (sell rallies).
Uses swing points from 6h data and 1d trend filter to avoid counter-trend trades.
Target: 50-150 total trades over 4 years (12-37/year).
"""

name = "6h_MarketStructure_1dTrend_Filter"
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
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Swing points on 6h data: swing low (lowest low in window), swing high (highest high)
    window = 5  # look for swings in 5-bar window (2 before, current, 2 after)
    swing_low = np.full(n, np.nan)
    swing_high = np.full(n, np.nan)
    
    half = window // 2
    for i in range(half, n - half):
        # Swing low: lowest low in window
        window_low = low[i-half:i+half+1]
        if low[i] == np.min(window_low):
            swing_low[i] = low[i]
        # Swing high: highest high in window
        window_high = high[i-half:i+half+1]
        if high[i] == np.max(window_high):
            swing_high[i] = high[i]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, half + 1)  # warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get most recent swing points (look back up to 20 bars)
        lookback = min(20, i)
        prev_swing_low = np.nan
        prev_swing_high = np.nan
        
        for j in range(i-1, max(i-lookback, 0)-1, -1):
            if not np.isnan(swing_low[j]):
                prev_swing_low = swing_low[j]
                break
        
        for j in range(i-1, max(i-lookback, 0)-1, -1):
            if not np.isnan(swing_high[j]):
                prev_swing_high = swing_high[j]
                break
        
        if position == 0:
            # Long: price makes higher low above prior swing low in uptrend
            if (not np.isnan(prev_swing_low) and 
                low[i] > prev_swing_low and 
                close[i] > ema50_1d_aligned[i] and  # uptrend filter
                ema50_1d_aligned[i] > ema50_1d_aligned[max(i-3, 0)]):  # trend confirmation
                signals[i] = 0.25
                position = 1
            # Short: price makes lower high below prior swing high in downtrend
            elif (not np.isnan(prev_swing_high) and 
                  high[i] < prev_swing_high and 
                  close[i] < ema50_1d_aligned[i] and  # downtrend filter
                  ema50_1d_aligned[i] < ema50_1d_aligned[max(i-3, 0)]):  # trend confirmation
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: trend reversal or failure to make new high
            if close[i] < ema50_1d_aligned[i] or high[i] <= prev_swing_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: trend reversal or failure to make new low
            if close[i] > ema50_1d_aligned[i] or low[i] >= prev_swing_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
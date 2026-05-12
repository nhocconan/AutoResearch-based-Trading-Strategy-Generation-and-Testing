#!/usr/bin/env python3
name = "6h_TurtleSoup_1dTrend_Reversal"
timeframe = "6h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Load daily data once for trend filter and daily range
    df_1d = get_htf_data(prices, '1d')
    
    # Daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Daily OHLC for range calculation (previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate previous day's range
    prev_range = high_1d - low_1d
    
    # Calculate 60-period (5-day) high/low for 6h timeframe (equivalent to Donchian)
    # Using 60 periods of 6h = 5 days
    high_60 = pd.Series(high).rolling(window=60, min_periods=60).max().values
    low_60 = pd.Series(low).rolling(window=60, min_periods=60).min().values
    
    # Turtle Soup fade conditions: false breakout of 5-day high/low
    # Long setup: price breaks below 5-day low but reverses back above it
    # Short setup: price breaks above 5-day high but reverses back below it
    
    # False break down below 5-day low
    false_break_down = (low < low_60) & (close > low_60)
    # False break up above 5-day high
    false_break_up = (high > high_60) & (close < high_60)
    
    # Align daily trend and range to 6h
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    prev_range_aligned = align_htf_to_ltf(prices, df_1d, prev_range)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # need enough data for 60-period lookback
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or np.isnan(prev_range_aligned[i]) or
            np.isnan(high_60[i]) or np.isnan(low_60[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: false break down + above daily EMA (bullish bias)
            if (false_break_down[i] and 
                close[i] > ema50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: false break up + below daily EMA (bearish bias)
            elif (false_break_up[i] and 
                  close[i] < ema50_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit conditions: reverse signal or stop at 2x daily range
            if (false_break_up[i] or 
                close[i] < (low_60[i] - prev_range_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions: reverse signal or stop at 2x daily range
            if (false_break_down[i] or 
                close[i] > (high_60[i] + prev_range_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
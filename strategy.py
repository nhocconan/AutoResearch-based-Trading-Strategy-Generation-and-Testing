#!/usr/bin/env python3
# 4H_Fibonacci_Retracement_Trend_Scalp
# Hypothesis: During established trends (1d EMA50), price retraces to Fibonacci levels (38.2%, 50%, 61.8%) 
# before resuming trend. Entry on retracement with volume confirmation, exit on trend reversal.
# Works in bull markets (buying dips in uptrend) and bear markets (selling rallies in downtrend) 
# by only trading in direction of daily trend. Uses 4H timeframe for optimal trade frequency.

name = "4H_Fibonacci_Retracement_Trend_Scalp"
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
    volume = prices['volume'].values
    
    # Get daily data for trend and swing calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate daily swing high/low for Fibonacci retracement
    # Use 20-day lookback for swing points
    lookback = 20
    swing_high = pd.Series(df_1d['high']).rolling(window=lookback, min_periods=lookback).max().values
    swing_low = pd.Series(df_1d['low']).rolling(window=lookback, min_periods=lookback).min().values
    
    swing_high_aligned = align_htf_to_ltf(prices, df_1d, swing_high)
    swing_low_aligned = align_htf_to_ltf(prices, df_1d, swing_low)
    
    # Calculate Fibonacci levels
    diff = swing_high_aligned - swing_low_aligned
    fib_382 = swing_low_aligned + 0.382 * diff
    fib_500 = swing_low_aligned + 0.500 * diff
    fib_618 = swing_low_aligned + 0.618 * diff
    
    # Volume confirmation (20-period MA on 4H = ~3.3 days)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need daily EMA50 (50), swing calculation (20), volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(swing_high_aligned[i]) or 
            np.isnan(swing_low_aligned[i]) or 
            np.isnan(fib_382[i]) or 
            np.isnan(fib_500[i]) or 
            np.isnan(fib_618[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: uptrend + price at Fibonacci support + volume
            # Allow small tolerance around Fib levels
            tolerance = 0.001 * close[i]  # 0.1% tolerance
            at_fib_support = (abs(close[i] - fib_382[i]) <= tolerance or 
                             abs(close[i] - fib_500[i]) <= tolerance or 
                             abs(close[i] - fib_618[i]) <= tolerance)
            
            if uptrend and at_fib_support and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + price at Fibonacci resistance + volume
            elif downtrend and at_fib_support and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price moves above Fibonacci resistance
            if not uptrend or close[i] > fib_618[i] + tolerance:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price moves below Fibonacci support
            if not downtrend or close[i] < fib_382[i] - tolerance:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
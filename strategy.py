#!/usr/bin/env python3
"""
1d_MACD_Trend_WeeklyFilter
Hypothesis: On 1d timeframe, use MACD histogram for momentum confirmation and weekly EMA trend filter to capture multi-day trends in both bull and bear markets. The strategy aims for 10-25 trades/year by requiring MACD crossovers aligned with weekly trend, reducing whipsaws and frequency. Weekly trend filter ensures we only trade in the direction of higher timeframe momentum, improving win rate and reducing false signals.
"""
name = "1d_MACD_Trend_WeeklyFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # MACD on daily timeframe
    ema_fast = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_slow = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    
    # Get weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(35, 50)  # MACD(26,9) + weekly EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if weekly trend data not ready
        if np.isnan(ema_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: MACD crosses above signal line + weekly uptrend
            if (macd_hist[i] > 0 and macd_hist[i-1] <= 0 and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: MACD crosses below signal line + weekly downtrend
            elif (macd_hist[i] < 0 and macd_hist[i-1] >= 0 and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: MACD crosses in opposite direction
            if (position == 1 and macd_hist[i] < 0 and macd_hist[i-1] >= 0) or \
               (position == -1 and macd_hist[i] > 0 and macd_hist[i-1] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals
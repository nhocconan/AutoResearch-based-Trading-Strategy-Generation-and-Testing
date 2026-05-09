#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Turtle Soup Strategy with 1d Trend Filter
# Turtle Soup is a reversal pattern where price briefly penetrates a recent high/low
# then reverses, trapping breakout traders. Works in both trending and ranging markets.
# 1d trend filter ensures we take reversals in the direction of higher timeframe trend.
# Target: 20-40 trades/year (80-160 over 4 years) to avoid excessive trading.
name = "6h_TurtleSoup_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 20-period high/low for Turtle Soup setup
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 50-period EMA on 1d for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h
    ema50_1d_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema50_1d_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Turtle Soup Long: price briefly breaks below 20-period low then reverses
            # Only take in uptrend (price above 1d EMA50)
            if (low[i] < low_20[i] and close[i] > low_20[i] and 
                close[i] > ema50_1d_6h[i]):
                signals[i] = 0.25
                position = 1
            # Turtle Soup Short: price briefly breaks above 20-period high then reverses
            # Only take in downtrend (price below 1d EMA50)
            elif (high[i] > high_20[i] and close[i] < high_20[i] and 
                  close[i] < ema50_1d_6h[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 20-period low or trend changes
            if low[i] < low_20[i] or close[i] < ema50_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above 20-period high or trend changes
            if high[i] > high_20[i] or close[i] > ema50_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
"""
1d_1w_1w_Return_Trend_Filter
Hypothesis: On 1d timeframe, buy when 1-week total return is positive and price is above 1d SMA50, sell when 1-week return is negative and price is below 1d SMA50. Uses weekly trend filter to avoid counter-trend trades. Target: 10-20 trades/year for low fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # === 1w Data (HTF for weekly return) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1-week total return: (current close - close 5 days ago) / close 5 days ago
    # Since 1w = 5 trading days, we need to look back 5 bars
    returns_1w = np.zeros_like(close_1w)
    returns_1w[5:] = (close_1w[5:] - close_1w[:-5]) / close_1w[:-5]
    
    # Align weekly return to daily
    returns_1w_aligned = align_htf_to_ltf(prices, df_1w, returns_1w)
    
    # 1d SMA50 for trend filter
    close_series = pd.Series(close)
    sma_50 = close_series.rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(returns_1w_aligned[i]) or 
            np.isnan(sma_50[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: positive weekly return and price above SMA50
            if returns_1w_aligned[i] > 0 and close[i] > sma_50[i]:
                signals[i] = 0.25
                position = 1
                continue
            # Short: negative weekly return and price below SMA50
            elif returns_1w_aligned[i] < 0 and close[i] < sma_50[i]:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal
        elif position == 1:
            # Exit when weekly turn negative or price below SMA50
            if returns_1w_aligned[i] < 0 or close[i] < sma_50[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when weekly turn positive or price above SMA50
            if returns_1w_aligned[i] > 0 or close[i] > sma_50[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_1w_Return_Trend_Filter"
timeframe = "1d"
leverage = 1.0
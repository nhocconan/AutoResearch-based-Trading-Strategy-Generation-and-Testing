#!/usr/bin/env python3
# 1d_1w_Camarilla_R1S1_TrendFollow_With_TrendFilter
# Hypothesis: Daily close above/below weekly R1/S1 with weekly EMA34 trend filter.
# In uptrends (price > weekly EMA34), long weekly R1 breakouts; in downtrends (price < weekly EMA34), short weekly S1 breakouts.
# Uses daily timeframe for signal generation and weekly for trend/context. Target: 10-25 trades/year.

name = "1d_1w_Camarilla_R1S1_TrendFollow_With_TrendFilter"
timeframe = "1d"
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
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # Calculate weekly pivot points
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = pivot_1w + (high_1w - low_1w) * 1.1 / 12
    s1_1w = pivot_1w - (high_1w - low_1w) * 1.1 / 12
    
    # Calculate weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly indicators to daily timeframe
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure EMA34 is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or np.isnan(ema34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > weekly EMA34 (uptrend) and breaks above weekly R1
            if close[i] > ema34_1w_aligned[i] and close[i] > r1_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < weekly EMA34 (downtrend) and breaks below weekly S1
            elif close[i] < ema34_1w_aligned[i] and close[i] < s1_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below weekly S1 (reversal signal) or trend changes
            if close[i] < s1_1w_aligned[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above weekly R1 (reversal signal) or trend changes
            if close[i] > r1_1w_aligned[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
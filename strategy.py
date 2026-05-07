#!/usr/bin/env python3
"""
1d_Weekly_Pivot_Breakout_Trend_Filter_v1
Hypothesis: On 1d timeframe, buy when price breaks above weekly pivot R1 with weekly uptrend (weekly close > weekly EMA20); sell when breaks below weekly pivot S1 with weekly downtrend (weekly close < weekly EMA20). Uses weekly EMA20 for trend filter to avoid whipsaws. Designed for low trade frequency (<25/year) to minimize fee drag and work in both bull and bear markets.
"""
name = "1d_Weekly_Pivot_Breakout_Trend_Filter_v1"
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
    
    # Get weekly data for pivot and trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Calculate weekly EMA20 for trend filter
    ema20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema20_weekly)
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    pivot_weekly = (high_weekly + low_weekly + close_weekly) / 3.0
    r1_weekly = 2 * pivot_weekly - low_weekly
    s1_weekly = 2 * pivot_weekly - high_weekly
    
    # Align weekly pivot levels to daily timeframe
    r1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, r1_weekly)
    s1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, s1_weekly)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient warmup for weekly EMA20
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema20_weekly_aligned[i]) or 
            np.isnan(r1_weekly_aligned[i]) or 
            np.isnan(s1_weekly_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R1 + weekly uptrend
            if (close[i] > r1_weekly_aligned[i] and 
                close[i] > ema20_weekly_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 + weekly downtrend
            elif (close[i] < s1_weekly_aligned[i] and 
                  close[i] < ema20_weekly_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to opposite weekly pivot level
            if position == 1:
                if close[i] < s1_weekly_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > r1_weekly_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
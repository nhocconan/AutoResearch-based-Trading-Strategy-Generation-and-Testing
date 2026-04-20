#!/usr/bin/env python3
"""
1d_WeeklyPivot_TrendFollowing
Hypothesis: Use weekly pivot points (PP, R1, S1) as support/resistance on daily timeframe.
Go long when price closes above weekly R1 with bullish weekly trend (price > weekly EMA50),
short when price closes below weekly S1 with bearish weekly trend (price < weekly EMA50).
Weekly trend filter ensures we trade with the higher-timeframe momentum, reducing whipsaws.
Position size 0.25 balances opportunity and risk. Designed to work in both bull and bear markets
by avoiding counter-trend trades. Target: 20-50 trades over 4 years (5-12/year).
"""

name = "1d_WeeklyPivot_TrendFollowing"
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
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    def ema(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            multiplier = 2.0 / (period + 1)
            result[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                result[i] = multiplier * values[i] + (1 - multiplier) * result[i-1]
        return result
    
    ema50_1w = ema(close_1w, 50)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate weekly pivot points (using previous week's OHLC)
    # Pivot Point (PP) = (High + Low + Close) / 3
    # R1 = 2*PP - Low
    # S1 = 2*PP - High
    high_shift = np.roll(high_1w, 1)
    low_shift = np.roll(low_1w, 1)
    close_shift = np.roll(close_1w, 1)
    # First week: use same values
    high_shift[0] = high_1w[0]
    low_shift[0] = low_1w[0]
    close_shift[0] = close_1w[0]
    
    pp = (high_shift + low_shift + close_shift) / 3.0
    r1 = 2 * pp - low_shift
    s1 = 2 * pp - high_shift
    
    # Align weekly pivot levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price closes above weekly R1 AND weekly uptrend (price > weekly EMA50)
            if close[i] > r1_aligned[i] and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price closes below weekly S1 AND weekly downtrend (price < weekly EMA50)
            elif close[i] < s1_aligned[i] and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below weekly PP OR weekly trend turns down
            if close[i] < pp[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above weekly PP OR weekly trend turns up
            if close[i] > pp[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
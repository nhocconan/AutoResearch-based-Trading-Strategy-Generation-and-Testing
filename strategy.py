#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_Camarilla_R1_S1_Breakout_TrendFilter_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (R1, S1) from previous week
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # Previous week's levels (to avoid look-ahead)
    pivot_w = (high_w[:-1] + low_w[:-1] + close_w[:-1]) / 3
    range_w = high_w[:-1] - low_w[:-1]
    R1_w = pivot_w + range_w * 1.1 / 12
    S1_w = pivot_w - range_w * 1.1 / 12
    
    # Align to daily (previous week's levels are known at daily open)
    R1_w_aligned = align_htf_to_ltf(prices, df_1w, R1_w)
    S1_w_aligned = align_htf_to_ltf(prices, df_1w, S1_w)
    
    # Weekly trend filter: EMA(21) on weekly close
    ema21_w = pd.Series(close_w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_w_aligned = align_htf_to_ltf(prices, df_1w, ema21_w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R1_w_aligned[i]) or np.isnan(S1_w_aligned[i]) or 
            np.isnan(ema21_w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R1 AND weekly close above EMA21 (uptrend)
            if close[i] > R1_w_aligned[i] and close_w_aligned[i] > ema21_w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 AND weekly close below EMA21 (downtrend)
            elif close[i] < S1_w_aligned[i] and close_w_aligned[i] < ema21_w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly S1 OR weekly close below EMA21
            if close[i] < S1_w_aligned[i] or close_w_aligned[i] < ema21_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above weekly R1 OR weekly close above EMA21
            if close[i] > R1_w_aligned[i] or close_w_aligned[i] > ema21_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly Camarilla levels (R1/S1) act as strong support/resistance.
# In uptrend (weekly close above EMA21), break above R1 signals continuation.
# In downtrend (weekly close below EMA21), break below S1 signals continuation.
# Trend filter avoids false breakouts in ranging markets.
# Weekly timeframe reduces whipsaw and captures major moves in both bull and bear markets.
# Target: 20-60 trades over 4 years (5-15/year) to minimize fee drag.
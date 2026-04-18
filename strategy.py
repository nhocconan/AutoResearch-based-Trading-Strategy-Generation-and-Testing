#!/usr/bin/env python3
"""
6h Weekly Pivot Breakout with Volume and EMA200 Filter
Hypothesis: Weekly pivot levels (from 1w data) act as strong support/resistance. Breaking above R1 or below S1 with volume confirmation and EMA200 trend filter captures momentum in both bull and bear markets. Weekly pivots are less noisy than daily and provide cleaner breakout signals. EMA200 filter ensures we only trade in the direction of the long-term trend, reducing whipsaws. Volume confirmation ensures breakouts are supported by participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_pivot_points(high, low, close):
    """Calculate standard pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H"""
    p = (high + low + close) / 3.0
    r1 = 2 * p - low
    s1 = 2 * p - high
    return p, r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points
    _, r1_1w, s1_1w = calculate_pivot_points(
        df_1w['high'].values,
        df_1w['low'].values,
        df_1w['close'].values
    )
    # Align to 6h timeframe (waits for weekly close)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # EMA200 on 1d for long-term trend filter (more stable than weekly EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Warmup for EMA200
    
    for i in range(start_idx, n):
        if np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        trend = ema200_1d_aligned[i]
        vol_ok = vol_confirm[i]
        r1 = r1_1w_aligned[i]
        s1 = s1_1w_aligned[i]
        
        if position == 0:
            # Enter long: price breaks above weekly R1 with volume + above EMA200 (uptrend)
            if (not np.isnan(r1) and 
                vol_ok and 
                close[i] > r1 and 
                close[i] > trend):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly S1 with volume + below EMA200 (downtrend)
            elif (not np.isnan(s1) and 
                  vol_ok and 
                  close[i] < s1 and 
                  close[i] < trend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below weekly S1 or below EMA200
            if (not np.isnan(s1) and close[i] < s1) or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above weekly R1 or above EMA200
            if (not np.isnan(r1) and close[i] > r1) or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Weekly_Pivot_Breakout_Volume_EMA200"
timeframe = "6h"
leverage = 1.0
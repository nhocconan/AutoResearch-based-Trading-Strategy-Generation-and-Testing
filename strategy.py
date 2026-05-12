#!/usr/bin/env python3
name = "6h_WeeklyPivot_Trend_Reversal"
timeframe = "6h"
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
    
    # Load weekly data once for pivot points and trend
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Weekly OHLC for pivot points (previous week)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly close for EMA trend filter
    close_1w_for_trend = df_1w['close'].values
    
    # Calculate Weekly Pivot Points (standard formula)
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    p = (high_1w + low_1w + close_1w) / 3
    r1 = 2 * p - low_1w
    s1 = 2 * p - high_1w
    
    # Align weekly pivot points to 6h (wait for weekly close)
    p_aligned = align_htf_to_ltf(prices, df_1w, p)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Weekly EMA(34) for trend filter
    ema34_1w = pd.Series(close_1w_for_trend).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(p_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above weekly pivot + weekly trend up + volume spike
            if (close[i] > p_aligned[i] and 
                close[i] > ema34_1w_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below weekly pivot + weekly trend down + volume spike
            elif (close[i] < p_aligned[i] and 
                  close[i] < ema34_1w_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below weekly pivot
            if close[i] < p_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above weekly pivot
            if close[i] > p_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
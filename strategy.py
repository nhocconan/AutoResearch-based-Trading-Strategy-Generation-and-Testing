#!/usr/bin/env python3
name = "6h_WeeklyPivot_Trend_Reversal_v2"
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
    
    # Load weekly data once for pivot calculation and trend filter
    df_w = get_htf_data(prices, '1w')
    
    # Weekly EMA(34) for trend filter
    close_w = df_w['close'].values
    ema34_w = pd.Series(close_w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_w_aligned = align_htf_to_ltf(prices, df_w, ema34_w)
    
    # Weekly OHLC for pivot points (previous week)
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Calculate weekly pivot and S1/R1 levels
    pivot = (high_w + low_w + close_w) / 3
    r1 = (2 * pivot) - low_w
    s1 = (2 * pivot) - high_w
    
    # Align weekly pivot levels to 6h (wait for weekly close)
    pivot_aligned = align_htf_to_ltf(prices, df_w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_w, s1)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema34_w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + weekly trend up + volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema34_w_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + weekly trend down + volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema34_w_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below pivot
            if close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above pivot
            if close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
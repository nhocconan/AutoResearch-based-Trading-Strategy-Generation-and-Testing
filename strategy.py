#!/usr/bin/env python3
"""
6h_WeeklyPivot_Breakout_1dTrend_Volume
Hypothesis: Use weekly (W) pivot points as structural support/resistance, with 1d EMA50 trend filter and volume confirmation. Weekly pivots are less noisy than daily and hold better in trends. Enter long when price breaks above weekly R1 with 1d uptrend and volume spike; short when breaks below weekly S1 with 1d downtrend and volume spike. Designed for low trade frequency (10-25/year) to avoid fee drag.
"""
name = "6h_WeeklyPivot_Breakout_1dTrend_Volume"
timeframe = "6h"
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
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly OHLC for pivot points
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Weekly pivot points (standard formula)
    pivot_w = (high_w + low_w + close_w) / 3
    range_w = high_w - low_w
    r1_w = pivot_w + (range_w * 1.0)  # R1 = P + (H-L)
    s1_w = pivot_w - (range_w * 1.0)  # S1 = P - (H-L)
    
    # Align weekly pivot levels to 6h timeframe
    r1_w_aligned = align_htf_to_ltf(prices, df_w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_w, s1_w)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need sufficient warmup for averages
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(r1_w_aligned[i]) or np.isnan(s1_w_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R1 + 1d uptrend + volume spike
            if (close[i] > r1_w_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 + 1d downtrend + volume spike
            elif (close[i] < s1_w_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to opposite weekly pivot level (S1 for long, R1 for short)
            if position == 1:
                if close[i] <= s1_w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] >= r1_w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
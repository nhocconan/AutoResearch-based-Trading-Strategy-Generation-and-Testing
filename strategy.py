#!/usr/bin/env python3
# 12h_1w_1d_Triple_Timeframe_Confluence
# Hypothesis: Trade breakouts of weekly pivot R1/S1 levels on 12h timeframe with daily trend filter and volume confirmation.
# Weekly pivots provide strong institutional levels; daily trend ensures alignment with higher timeframe momentum.
# Volume spike confirms institutional participation. Works in bull/bear by trading breakouts in direction of daily trend.
# Targets 15-30 trades per year by requiring confluence of weekly level, daily trend, and volume spike.

name = "12h_1w_1d_Triple_Timeframe_Confluence"
timeframe = "12h"
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
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly R1 and S1 levels using previous week's data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point and range
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Weekly R1 and S1 levels
    s1_1w = pivot_1w - range_1w * 1.0
    r1_1w = pivot_1w + range_1w * 1.0
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly levels and daily EMA to 12h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above weekly R1, daily uptrend, volume spike
            if (close[i] > r1_aligned[i] * 1.002 and 
                close[i] > ema_34_aligned[i] and
                volume[i] > 2.0 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly S1, daily downtrend, volume spike
            elif (close[i] < s1_aligned[i] * 0.998 and 
                  close[i] < ema_34_aligned[i] and
                  volume[i] > 2.0 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below weekly S1 or trend reversal
            if close[i] < s1_aligned[i] * 0.998 or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above weekly R1 or trend reversal
            if close[i] > r1_aligned[i] * 1.002 or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
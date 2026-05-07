#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1wTrend_Volume
Hypothesis: Price breaking above Camarilla R1 or below S1 levels from prior 1w period,
combined with 1w EMA50 trend filter and volume confirmation, captures momentum moves
while reducing false signals in choppy conditions. Weekly trend filter ensures alignment
with higher timeframe momentum, improving performance in both bull and bear markets.
"""
name = "12h_Camarilla_R1S1_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Camarilla levels and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w Camarilla R1/S1 levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    range_1w = high_1w - low_1w
    r1_1w = close_1w + 1.1666 * range_1w * 1.1 / 2
    s1_1w = close_1w - 1.1666 * range_1w * 1.1 / 2
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all to 12h timeframe
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + 1w uptrend + volume
            if close[i] > r1_1w_aligned[i] and close[i] > ema_50_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S1 + 1w downtrend + volume
            elif close[i] < s1_1w_aligned[i] and close[i] < ema_50_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.30
                position = -1
        elif position != 0:
            # Exit: price crosses back through the opposite S1/R1 level
            if position == 1:
                if close[i] < s1_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.30
            else:  # position == -1
                if close[i] > r1_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.30
    
    return signals
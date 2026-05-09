#!/usr/bin/env python3
name = "6H_WeeklyPivot_Trend_Filter_With_Volume"
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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot levels (using previous week's data)
    prev_high_w = df_1w['high'].shift(1).values
    prev_low_w = df_1w['low'].shift(1).values
    prev_close_w = df_1w['close'].shift(1).values
    
    # Weekly pivot: P = (H + L + C) / 3
    pivot_w = (prev_high_w + prev_low_w + prev_close_w) / 3.0
    # Weekly R1 and S1
    r1_w = prev_close_w + (prev_high_w - prev_low_w) * 1.1 / 12.0
    s1_w = prev_close_w - (prev_high_w - prev_low_w) * 1.1 / 12.0
    
    # Align weekly pivot levels to 6h timeframe
    pivot_w_6h = align_htf_to_ltf(prices, df_1w, pivot_w)
    r1_w_6h = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_w_6h = align_htf_to_ltf(prices, df_1w, s1_w)
    
    # Daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if EMA data not ready
        if np.isnan(ema50_1d_6h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if weekly pivot levels not ready
        if np.isnan(r1_w_6h[i]) or np.isnan(s1_w_6h[i]) or np.isnan(pivot_w_6h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above weekly pivot AND above weekly R1 + volume confirmation + price above daily EMA50
            if close[i] > pivot_w_6h[i] and close[i] > r1_w_6h[i] and volume_confirm[i] and close[i] > ema50_1d_6h[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price below weekly pivot AND below weekly S1 + volume confirmation + price below daily EMA50
            elif close[i] < pivot_w_6h[i] and close[i] < s1_w_6h[i] and volume_confirm[i] and close[i] < ema50_1d_6h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price drops below weekly pivot
            if close[i] < pivot_w_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above weekly pivot
            if close[i] > pivot_w_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
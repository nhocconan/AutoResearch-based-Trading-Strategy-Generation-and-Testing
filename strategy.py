#!/usr/bin/env python3
name = "1D_Weekly_Camarilla_R1S1_Breakout_WeeklyTrend"
timeframe = "1d"
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
    
    # Get weekly data for Camarilla pivot levels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    # Weekly pivot point and Camarilla levels (R1, S1)
    pivot_w = (high_w + low_w + close_w) / 3
    range_w = high_w - low_w
    r1_w = pivot_w + (range_w * 1.1 / 12)
    s1_w = pivot_w - (range_w * 1.1 / 12)
    
    # Weekly trend filter: close above/below 21-period EMA
    ema21_w = pd.Series(close_w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align weekly levels to daily
    r1_w_aligned = align_htf_to_ltf(prices, df_weekly, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_weekly, s1_w)
    ema21_w_aligned = align_htf_to_ltf(prices, df_weekly, ema21_w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(r1_w_aligned[i]) or np.isnan(s1_w_aligned[i]) or np.isnan(ema21_w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above weekly R1 + above weekly EMA21 + volume confirmation
            if close[i] > r1_w_aligned[i] and close[i] > ema21_w_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly S1 + below weekly EMA21 + volume confirmation
            elif close[i] < s1_w_aligned[i] and close[i] < ema21_w_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below weekly EMA21 (trend change)
            if close[i] < ema21_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above weekly EMA21 (trend change)
            if close[i] > ema21_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
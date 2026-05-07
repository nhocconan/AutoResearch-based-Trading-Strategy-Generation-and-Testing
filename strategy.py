#!/usr/bin/env python3
name = "6h_WeeklyPivot_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    # 1d EMA26 trend filter
    ema_26_1d = pd.Series(df_1d['close']).ewm(span=26, adjust=False, min_periods=26).mean().values
    ema_26_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_26_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (vol_ma * 1.5)
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (R1, S1) from previous week
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    range_hl = prev_week_high - prev_week_low
    r1 = pivot + (range_hl * 1.1 / 4)   # R1 level (weekly)
    s1 = pivot - (range_hl * 1.1 / 4)   # S1 level (weekly)
    
    # Align weekly pivot levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 26  # Wait for EMA26 and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema_26_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above weekly R1 + above 1d EMA26 + volume spike
            if close[i] > r1_aligned[i] and close[i] > ema_26_1d_aligned[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S1 + below 1d EMA26 + volume spike
            elif close[i] < s1_aligned[i] and close[i] < ema_26_1d_aligned[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to opposite weekly level or breaks in opposite direction
            if position == 1:
                if close[i] < s1_aligned[i] or close[i] < ema_26_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > r1_aligned[i] or close[i] > ema_26_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
name = "1d_WeeklyPivot_DailyTrend_Volume"
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
    
    # Load 1w data once for weekly pivots
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly OHLC for pivot calculation (previous week)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (R1, S1)
    pivot = (high_1w + low_1w + close_1w) / 3
    r1 = pivot * 2 - low_1w  # R1 = 2*P - Low
    s1 = pivot * 2 - high_1w # S1 = 2*P - High
    
    # Align weekly pivots to daily (wait for weekly close)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Daily EMA34 for trend filter
    ema_34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume spike: current volume > 2.5x 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # ensure EMA has enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + above EMA34 + volume spike
            if (close[i] > r1_aligned[i] and close[i] > ema_34[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + below EMA34 + volume spike
            elif (close[i] < s1_aligned[i] and close[i] < ema_34[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below S1
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above R1
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
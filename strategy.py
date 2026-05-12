#!/usr/bin/env python3
name = "1h_Camarilla_R1S1_Breakout_4hTrend_Volume"
timeframe = "1h"
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
    
    # Calculate Camarilla pivot levels from 4h data
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Camarilla pivot calculation: P = (H+L+C)/3, Range = H-L
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    r1_4h = pivot_4h + (range_4h * 1.0 / 12.0)
    s1_4h = pivot_4h - (range_4h * 1.0 / 12.0)
    
    # Align 4h Camarilla levels to 1h
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (2.0 * vol_avg)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: breakout above R1 + above 4h EMA50 + volume filter
            if high[i] > r1_4h_aligned[i] and close[i] > ema_50_4h_aligned[i] and vol_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: breakdown below S1 + below 4h EMA50 + volume filter
            elif low[i] < s1_4h_aligned[i] and close[i] < ema_50_4h_aligned[i] and vol_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: breakdown below S1 or below 4h EMA50
            if low[i] < s1_4h_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: breakout above R1 or above 4h EMA50
            if high[i] > r1_4h_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
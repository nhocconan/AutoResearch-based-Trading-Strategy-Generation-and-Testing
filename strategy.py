#!/usr/bin/env python3
name = "4h_Camarilla_R1S1_Breakout_1dTrend_Volume_Enhanced"
timeframe = "4h"
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
    
    # Load 1d data for trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla R1 and S1 for previous day
    p = (high_1d + low_1d + close_1d) / 3
    r1 = p + (high_1d - low_1d) * 1.1 / 12
    s1 = p - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 4h (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike: current volume > 2.2x 25-period average (stricter)
    vol_avg = pd.Series(volume).rolling(window=25, min_periods=25).mean().values
    vol_spike = volume > (2.2 * vol_avg)
    
    # Momentum filter: price > 5-period EMA for long, < 5-period EMA for short
    ema_5 = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Add hysteresis: require price to be outside pivot band for 3 consecutive bars (more stringent)
    outside_long = np.zeros(n, dtype=bool)
    outside_short = np.zeros(n, dtype=bool)
    
    for i in range(2, n):  # start from index 2 to check 3-bar condition
        outside_long[i] = (close[i] > r1_aligned[i] and 
                          close[i-1] > r1_aligned[i-1] and 
                          close[i-2] > r1_aligned[i-2])
        outside_short[i] = (close[i] < s1_aligned[i] and 
                           close[i-1] < s1_aligned[i-1] and 
                           close[i-2] < s1_aligned[i-2])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_spike[i]) or
            np.isnan(ema_5[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price outside R1 band for 3 bars + above 1d EMA34 + volume spike + above EMA5
            if (outside_long[i] and close[i] > ema_34_1d_aligned[i] and 
                vol_spike[i] and close[i] > ema_5[i]):
                signals[i] = 0.25
                position = 1
            # Short: price outside S1 band for 3 bars + below 1d EMA34 + volume spike + below EMA5
            elif (outside_short[i] and close[i] < ema_34_1d_aligned[i] and 
                  vol_spike[i] and close[i] < ema_5[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below S1 (more responsive exit)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above R1 (more responsive exit)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
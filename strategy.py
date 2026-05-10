#!/usr/bin/env python3
"""
1h_4h1d_Camarilla_R1S1_Breakout_TrendFilter_v1
Hypothesis: Use 4h for signal direction via EMA50 trend filter and 1d for Camarilla pivot calculation.
Enter long in uptrend when price breaks above daily R1 with volume confirmation, short in downtrend when price breaks below daily S1.
Use 1h only for precise entry timing to reduce false breakouts. Target 15-30 trades/year.
Works in bull via trend-following breaks, in bear via mean-reversion at strong daily levels.
"""

name = "1h_4h1d_Camarilla_R1S1_Breakout_TrendFilter_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla R1 and S1 calculation
    camarilla_range = high_prev - low_prev
    camarilla_r1 = close_prev + camarilla_range * 1.1 / 12
    camarilla_s1 = close_prev - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Price and volume arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 24-period EMA (1 day of 1h bars)
    vol_ema24 = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values
    volume_filter = volume > vol_ema24 * 1.5
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50 and previous day data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 4h trend: price vs EMA50
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # Long: uptrend AND price breaks above daily R1 with volume and session
            if uptrend and high[i] > camarilla_r1_aligned[i] and volume_filter[i] and session_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: downtrend AND price breaks below daily S1 with volume and session
            elif downtrend and low[i] < camarilla_s1_aligned[i] and volume_filter[i] and session_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below daily S1 OR trend changes to downtrend
            if low[i] < camarilla_s1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above daily R1 OR trend changes to uptrend
            if high[i] > camarilla_r1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
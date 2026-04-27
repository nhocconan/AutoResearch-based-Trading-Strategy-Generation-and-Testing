#!/usr/bin/env python3
"""
#100994 - 1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
Hypothesis: Combine Camarilla pivot levels from 1d with 4h EMA50 trend filter and volume confirmation on 1h timeframe.
Long when price breaks above R1 with 4h uptrend and volume spike; short when breaks below S1 with 4h downtrend and volume spike.
Uses 1d for pivot calculation (more stable), 4h for trend filter, 1h for entry timing. Designed for 15-30 trades/year to minimize fee drag.
Works in both bull and bear markets by following 4h trend direction.
"""

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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for today using yesterday's data
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_1d = (high_1d[:-1] + low_1d[:-1] + close_1d[:-1]) / 3
    range_1d = high_1d[:-1] - low_1d[:-1]
    r1_1d = close_1d[:-1] + range_1d * 1.1 / 12
    s1_1d = close_1d[:-1] - range_1d * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume filter: volume > 2.0x 24-period average (more selective)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    # Session filter: 08-20 UTC (reduce noise trades)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check session filter
        if not session_filter[i]:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Long condition: price breaks above R1, above 4h EMA50, volume spike, during session
        if (close[i] > r1_1d_aligned[i] and 
            close[i] > ema50_4h_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.20
            position = 1
        # Short condition: price breaks below S1, below 4h EMA50, volume spike, during session
        elif (close[i] < s1_1d_aligned[i] and 
              close[i] < ema50_4h_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.20
            position = -1
        # Exit conditions: price returns to opposite Camarilla level
        elif position == 1 and close[i] < s1_1d_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > r1_1d_aligned[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0
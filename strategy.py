#!/usr/bin/env python3
name = "1H_Camarilla_R1S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day Camarilla pivot levels (R1, S1)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels (R1, S1)
    r1_1d = pivot_1d + (range_1d * 1.1 / 12)
    s1_1d = pivot_1d - (range_1d * 1.1 / 12)
    
    # Align to 1h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 4h EMA20 for trend filter
    ema20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Daily volume confirmation: current volume > 1.5x 20-day average
    volume_avg_20d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_avg_20d_aligned = align_htf_to_ltf(prices, df_1d, volume_avg_20d)
    volume_confirm = volume > (volume_avg_20d_aligned * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema20_4h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R1 + above 4h EMA20 + volume confirmation
            if close[i] > r1_aligned[i] and close[i] > ema20_4h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below S1 + below 4h EMA20 + volume confirmation
            elif close[i] < s1_aligned[i] and close[i] < ema20_4h_aligned[i] and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price below 4h EMA20 (trend change)
            if close[i] < ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price above 4h EMA20 (trend change)
            if close[i] > ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
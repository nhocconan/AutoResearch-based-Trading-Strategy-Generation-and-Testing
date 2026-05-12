#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS"
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
    
    # === 12h Data for trend (HTF) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # === 12h EMA50 for trend ===
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # === 1d Data for Camarilla pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 1d Camarilla pivot levels (R1, S1) ===
    close_prev = np.roll(close_1d, 1)
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    range_1d = high_prev - low_prev
    r1 = close_prev + (range_1d * 1.0 / 12)
    s1 = close_prev - (range_1d * 1.0 / 12)
    
    # Align to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Volume spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close breaks above R1 + volume spike + price above 12h EMA
            if (close[i] > r1_aligned[i] and 
                volume_spike[i] and
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S1 + volume spike + price below 12h EMA
            elif (close[i] < s1_aligned[i] and 
                  volume_spike[i] and
                  close[i] < ema50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close below S1 or price below EMA
            if close[i] < s1_aligned[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close above R1 or price above EMA
            if close[i] > r1_aligned[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
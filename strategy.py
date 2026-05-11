#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Breakouts at Camarilla R1/S1 levels (from previous day) with 1d EMA trend filter and volume confirmation.
This strategy trades breakouts in the direction of the 1d trend, using the previous day's Camarilla levels
as breakout points. Works in both bull and bear markets by aligning with higher timeframe trend.
Volume confirmation filters out false breakouts. Low trade frequency reduces fee drag.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # === 1d Data for Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    # === 1d Data for Camarilla Levels (previous day's OHLC) ===
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ph_1d = df_1d['high'].values
    pl_1d = df_1d['low'].values
    pc_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R1, S1
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    rang = ph_1d - pl_1d
    r1 = pc_1d + rang * 1.1 / 12
    s1 = pc_1d - rang * 1.1 / 12
    
    # Align to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Volume Filter (1.5x 20-period EMA on 12h) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers 1d EMA and daily data)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price closes above R1 with uptrend and volume
            if (close[i] > r1_aligned[i] and 
                close[i] > ema34_aligned[i] and 
                volume_ok[i]):
                signals[i] = 0.30
                position = 1
            # Short breakdown: price closes below S1 with downtrend and volume
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema34_aligned[i] and 
                  volume_ok[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price closes below S1 (mean reversion)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30  # maintain position
        elif position == -1:
            # Short exit: price closes above R1 (mean reversion)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30  # maintain position
    
    return signals
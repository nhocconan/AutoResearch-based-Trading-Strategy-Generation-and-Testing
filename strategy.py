#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Breakout_1wTrend_Volume_Confirmation
Hypothesis: Use weekly trend filter with daily Camarilla pivot breakouts on 12h timeframe.
Weekly trend ensures alignment with major market direction, reducing false breakouts.
Volume confirmation adds conviction. Designed for fewer trades (target: 50-150/4 years) to minimize fee drag.
Works in bull/bear by following weekly trend direction.
"""

name = "12h_Camarilla_Pivot_Breakout_1wTrend_Volume_Confirmation"
timeframe = "12h"
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
    
    # === Weekly Trend Filter (EMA34) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # === Daily OHLC for Camarilla Pivots ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    ph = df_1d['high'].values
    pl = df_1d['low'].values
    pc = df_1d['close'].values
    
    # Camarilla levels: R3, R2, S2, S3
    camarilla_r3 = pc + (ph - pl) * 1.1 / 4
    camarilla_r2 = pc + (ph - pl) * 1.1 / 6
    camarilla_s2 = pc - (ph - pl) * 1.1 / 6
    camarilla_s3 = pc - (ph - pl) * 1.1 / 4
    
    # Align to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    r2_12h = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    s2_12h = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    s3_12h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === Volume Filter (1.5x 20-period EMA on 12h) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers weekly and daily calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_12h[i]) or np.isnan(r2_12h[i]) or np.isnan(s2_12h[i]) or np.isnan(s3_12h[i]) or
            np.isnan(ema34_12h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R3 with weekly uptrend and volume
            if (close[i] > r3_12h[i] and 
                close[i] > ema34_12h[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S3 with weekly downtrend and volume
            elif (close[i] < s3_12h[i] and 
                  close[i] < ema34_12h[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below R2 (profit target or reversal)
            if close[i] < r2_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price breaks above S2 (profit target or reversal)
            if close[i] > s2_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals
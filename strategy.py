#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_1wTrend_Volume
Hypothesis: Trade breakouts at daily Camarilla R1/S1 levels on 1d timeframe with weekly trend filter and volume confirmation.
Works in both bull and bear markets by aligning with the weekly trend direction.
Targets 30-100 trades total over 4 years (7-25/year) by using tight entry conditions.
"""

name = "1d_Camarilla_R1_S1_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly OHLC for Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA34 for trend
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # === Daily OHLC for Camarilla Pivots ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    ph = df_1d['high'].values
    pl = df_1d['low'].values
    pc = df_1d['close'].values
    
    # Camarilla R1/S1 (most significant levels for breakout)
    camarilla_r1 = pc + (ph - pl) * 1.1 / 2
    camarilla_s1 = pc - (ph - pl) * 1.1 / 2
    
    # Align to daily timeframe
    r1_1d = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_1d = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === Volume Filter (2.0x 20-period EMA on daily) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers weekly and daily calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_1d[i]) or np.isnan(s1_1d[i]) or np.isnan(ema34_1d[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price closes above R1 with uptrend and volume
            if (close[i] > r1_1d[i] and 
                close[i] > ema34_1d[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price closes below S1 with downtrend and volume
            elif (close[i] < s1_1d[i] and 
                  close[i] < ema34_1d[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below S1 (reversal)
            if close[i] < s1_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price closes above R1 (reversal)
            if close[i] > r1_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals
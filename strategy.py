#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_Volume
Hypothesis: Breakouts at Camarilla R1 (resistance) and S1 (support) with 12h EMA50 trend filter and volume confirmation.
Works in both bull and bear markets by aligning with higher timeframe trend. Camarilla levels provide statistically
relevant support/resistance, while volume filters false breakouts. Low trade frequency reduces fee drag.
"""

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_Volume"
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
    
    # === 12h Data for Trend Filter (EMA50) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # === Daily Data for Camarilla Pivot Levels (R1, S1) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC
    ph_1d = df_1d['high'].values
    pl_1d = df_1d['low'].values
    pc_1d = df_1d['close'].values
    
    # Camarilla pivot levels
    # R1 = C + (H-L) * 1.1/12
    # S1 = C - (H-L) * 1.1/12
    camarilla_r1 = pc_1d + (ph_1d - pl_1d) * 1.1 / 12
    camarilla_s1 = pc_1d - (ph_1d - pl_1d) * 1.1 / 12
    
    # Align to 4h timeframe
    camarilla_r1_4h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_4h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === Volume Filter (1.5x 20-period EMA on 4h) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers 12h EMA50 and daily data)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r1_4h[i]) or np.isnan(camarilla_s1_4h[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price closes above Camarilla R1 with uptrend and volume
            if (close[i] > camarilla_r1_4h[i] and 
                close[i] > ema50_12h_aligned[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price closes below Camarilla S1 with downtrend and volume
            elif (close[i] < camarilla_s1_4h[i] and 
                  close[i] < ema50_12h_aligned[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below Camarilla S1 (mean reversion)
            if close[i] < camarilla_s1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price closes above Camarilla R1 (mean reversion)
            if close[i] > camarilla_r1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals
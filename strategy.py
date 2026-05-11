#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_Volume
Hypothesis: Trade breakouts at Camarilla R1/S1 levels on daily timeframe with weekly trend filter and volume confirmation.
This strategy uses a higher timeframe (weekly) to determine trend direction, reducing whipsaw and capturing stronger moves.
By focusing on daily breakouts with weekly trend alignment, it aims to work in both bull and bear markets while maintaining
low trade frequency to minimize fee drag.
"""

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "1d"
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
    
    # === Weekly Data for Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # === Daily Data for Camarilla Pivots (using previous day) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    ph_1d = df_1d['high'].values
    pl_1d = df_1d['low'].values
    pc_1d = df_1d['close'].values
    
    # Camarilla levels (R1, S1 are most significant for breakouts)
    # R1 = C + (H-L) * 1.1/12
    # S1 = C - (H-L) * 1.1/12
    r1_1d = pc_1d + (ph_1d - pl_1d) * 1.1 / 12
    s1_1d = pc_1d - (ph_1d - pl_1d) * 1.1 / 12
    
    # Align to daily timeframe (same as price data)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === Volume Filter (2.0x 20-period EMA on daily) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers weekly EMA and daily pivots)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price closes above R1 with weekly uptrend and volume
            if (close[i] > r1_1d_aligned[i] and 
                close[i] > ema20_1w_aligned[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price closes below S1 with weekly downtrend and volume
            elif (close[i] < s1_1d_aligned[i] and 
                  close[i] < ema20_1w_aligned[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below S1 (reversion to mean)
            if close[i] < s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price closes above R1 (reversion to mean)
            if close[i] > r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals
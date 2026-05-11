#!/usr/bin/env python3
"""
1h_4h1d_Camarilla_R1S1_Breakout
Hypothesis: Use daily Camarilla R1/S1 levels for breakout direction on 1h timeframe, filtered by 4h trend (EMA50) and volume confirmation.
Targets 15-37 trades/year by using higher timeframe for signal direction and 1h only for precise entry timing.
Works in bull/bear markets by aligning with 4h trend direction.
"""

name = "1h_4h1d_Camarilla_R1S1_Breakout"
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
    
    # Align to 1h timeframe
    r1_1h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_1h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === 4h Trend Filter (EMA50) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1h = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # === Volume Filter (1.5x 20-period EMA on 1h) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # === Session Filter: 08-20 UTC ===
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_ok = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers daily and 4h calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(r1_1h[i]) or np.isnan(s1_1h[i]) or np.isnan(ema50_1h[i]) or 
            np.isnan(volume_ok[i]) or not session_ok[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R1 with uptrend and volume
            if (close[i] > r1_1h[i] and 
                close[i] > ema50_1h[i] and 
                volume_ok[i]):
                signals[i] = 0.20
                position = 1
            # Short breakdown: price breaks below S1 with downtrend and volume
            elif (close[i] < s1_1h[i] and 
                  close[i] < ema50_1h[i] and 
                  volume_ok[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 (reversal)
            if close[i] < s1_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20  # maintain position
        elif position == -1:
            # Short exit: price breaks above R1 (reversal)
            if close[i] > r1_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20  # maintain position
    
    return signals
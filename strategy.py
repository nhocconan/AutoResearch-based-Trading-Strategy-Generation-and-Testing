#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
Hypothesis: Trade breakouts at Camarilla R1/S1 levels on 1h timeframe with 4h EMA trend filter and volume confirmation.
4h trend provides directional bias, 1h provides entry timing precision. Volume confirms breakout strength.
Works in bull/bear by aligning with 4h trend. Target: 15-37 trades/year.
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
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
    
    # === 4h Data for Trend Filter ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # === Daily Data for Camarilla Pivots (using previous day) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    ph_1d = df_1d['high'].values
    pl_1d = df_1d['low'].values
    pc_1d = df_1d['close'].values
    
    # Camarilla levels (R1, S1 are most significant for breakouts)
    r1_1d = pc_1d + (ph_1d - pl_1d) * 1.1 / 12
    s1_1d = pc_1d - (ph_1d - pl_1d) * 1.1 / 12
    
    # Align to 1h timeframe
    r1_1h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === Volume Filter (2.0x 24-period EMA on 1h) ===
    vol_ema24 = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values
    volume_ok = volume > vol_ema24 * 2.0
    
    # === Session Filter: 08-20 UTC ===
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_ok = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers 4h EMA and daily pivots)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_1h[i]) or np.isnan(s1_1h[i]) or 
            np.isnan(ema21_4h_aligned[i]) or np.isnan(volume_ok[i]) or 
            np.isnan(session_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if not session_ok[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price closes above R1 with uptrend and volume
            if (close[i] > r1_1h[i] and 
                close[i] > ema21_4h_aligned[i] and 
                volume_ok[i]):
                signals[i] = 0.20
                position = 1
            # Short breakdown: price closes below S1 with downtrend and volume
            elif (close[i] < s1_1h[i] and 
                  close[i] < ema21_4h_aligned[i] and 
                  volume_ok[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price closes below S1 (reversion to mean)
            if close[i] < s1_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20  # maintain position
        elif position == -1:
            # Short exit: price closes above R1 (reversion to mean)
            if close[i] > r1_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20  # maintain position
    
    return signals
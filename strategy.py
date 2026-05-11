#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_4hTrend_Volume
Hypothesis: Trade breakouts at 4-hour Camarilla R1/S1 levels with 4-hour trend filter (EMA34) and volume confirmation on 1h timeframe. Uses 4h for signal direction and 1h for entry timing to achieve 15-37 trades/year. Works in bull/bear by aligning with 4h trend.
"""

name = "1h_Camarilla_R1_S1_4hTrend_Volume"
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
    
    # === 4h OHLC for Camarilla Pivots ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 4h bar's OHLC
    ph = df_4h['high'].values
    pl = df_4h['low'].values
    pc = df_4h['close'].values
    
    # Camarilla R1/S1 (most significant levels for breakout)
    camarilla_r1 = pc + (ph - pl) * 1.1 / 2
    camarilla_s1 = pc - (ph - pl) * 1.1 / 2
    
    # Align to 1h timeframe
    r1_1h = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    s1_1h = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # === 4h Trend Filter (EMA34) ===
    ema34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1h = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # === Volume Filter (1.5x 20-period EMA on 1h) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # === Session Filter (08-20 UTC) ===
    # Pre-compute hours from datetime index
    hours = prices.index.hour
    session_ok = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers 4h calculations)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(r1_1h[i]) or np.isnan(s1_1h[i]) or np.isnan(ema34_1h[i]) or 
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
                close[i] > ema34_1h[i] and 
                volume_ok[i]):
                signals[i] = 0.20
                position = 1
            # Short breakdown: price breaks below S1 with downtrend and volume
            elif (close[i] < s1_1h[i] and 
                  close[i] < ema34_1h[i] and 
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
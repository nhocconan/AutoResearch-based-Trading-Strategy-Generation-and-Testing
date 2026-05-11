#!/usr/bin/env python3
"""
1h_4h1d_Camarilla_R1S1_Breakout_Volume
Hypothesis: Trade breakouts at daily Camarilla R1/S1 levels on 1h timeframe with 4h trend filter and volume confirmation.
Camarilla levels provide precise intraday support/resistance. Breakouts in direction of 4h trend with volume
should capture momentum moves while avoiding false breakouts. Uses 1h only for entry timing, 4h for trend direction.
"""

name = "1h_4h1d_Camarilla_R1S1_Breakout_Volume"
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
    
    # === Daily Camarilla Pivot Points (R1, S1) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ph_1d = df_1d['high'].values
    pl_1d = df_1d['low'].values
    pc_1d = df_1d['close'].values
    
    # Calculate daily pivot point
    pp_1d = (ph_1d + pl_1d + pc_1d) / 3.0
    # Camarilla R1 and S1 (inner support/resistance)
    r1_1d = pp_1d + 1.1 * (ph_1d - pl_1d) / 12
    s1_1d = pp_1d - 1.1 * (ph_1d - pl_1d) / 12
    
    # Align to 1h timeframe
    r1_1h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 4h Trend Filter (EMA34) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    ema34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1h = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # === Volume Filter (1.5x 20-period EMA on 1h) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_1h[i]) or np.isnan(s1_1h[i]) or np.isnan(ema34_1h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price closes above R1 with uptrend and volume
            if (close[i] > r1_1h[i] and 
                close[i] > ema34_1h[i] and 
                volume_ok[i]):
                signals[i] = 0.20
                position = 1
            # Short breakdown: price closes below S1 with downtrend and volume
            elif (close[i] < s1_1h[i] and 
                  close[i] < ema34_1h[i] and 
                  volume_ok[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price closes below S1 (mean reversion to support)
            if close[i] < s1_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20  # maintain position
        elif position == -1:
            # Short exit: price closes above R1 (mean reversion to resistance)
            if close[i] > r1_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20  # maintain position
    
    return signals
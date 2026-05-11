#!/usr/bin/env python3
"""
6h_1wR4S4_Breakout_1dTrend_Volume
Hypothesis: Trade breakouts at weekly R4/S4 levels on 6h timeframe with daily trend filter and volume confirmation.
Weekly R4/S4 represent strong support/resistance levels. Breakouts in direction of daily trend with volume
should capture significant moves. Uses weekly pivot calculation for fewer, more significant levels.
Designed to work in both bull and bear markets by aligning with daily trend direction.
"""

name = "6h_1wR4S4_Breakout_1dTrend_Volume"
timeframe = "6h"
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
    
    # === Weekly OHLC for Pivot Points ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Weekly Pivot Points from previous week's OHLC
    ph_w = df_1w['high'].values
    pl_w = df_1w['low'].values
    pc_w = df_1w['close'].values
    
    # Weekly Pivot Point (PP)
    pp_w = (ph_w + pl_w + pc_w) / 3.0
    # Weekly R4 and S4 (strongest breakout levels)
    r4_w = pp_w + 3 * (ph_w - pl_w)
    s4_w = pp_w - 3 * (ph_w - pl_w)
    
    # Align to 6h timeframe
    r4_6h = align_htf_to_ltf(prices, df_1w, r4_w)
    s4_6h = align_htf_to_ltf(prices, df_1w, s4_w)
    
    # === Daily Trend Filter (EMA34) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_6h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Volume Filter (2.0x 20-period EMA on 6h) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers weekly calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or np.isnan(ema34_6h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price closes above R4 with uptrend and volume
            if (close[i] > r4_6h[i] and 
                close[i] > ema34_6h[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price closes below S4 with downtrend and volume
            elif (close[i] < s4_6h[i] and 
                  close[i] < ema34_6h[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below midpoint between R4 and S4
            if close[i] < (r4_6h[i] + s4_6h[i]) / 2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price closes above midpoint between R4 and S4
            if close[i] > (r4_6h[i] + s4_6h[i]) / 2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals
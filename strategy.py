#!/usr/bin/env python3
"""
12h_Pivot_R4S4_Breakout_1dTrend_Volume
Hypothesis: Trade breakouts at daily pivot levels (R4/S4) on 12h timeframe with 1d trend filter and volume confirmation.
Daily pivot points (R4/S4) represent strong support/resistance levels. Breakouts in direction of daily trend with volume
confirmation should capture significant moves while avoiding excessive trading. Works in bull/bear markets by aligning
with daily trend direction. Targets 50-150 trades over 4 years to minimize fee drag.
"""

name = "12h_Pivot_R4S4_Breakout_1dTrend_Volume"
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
    
    # === Daily OHLC for Pivot Points ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Daily Pivot Points from previous day's OHLC
    ph_d = df_1d['high'].values
    pl_d = df_1d['low'].values
    pc_d = df_1d['close'].values
    
    # Daily Pivot Point (PP)
    pp_d = (ph_d + pl_d + pc_d) / 3.0
    # Daily R4 and S4 (strongest breakout levels)
    r4_d = pp_d + 3 * (ph_d - pl_d)
    s4_d = pp_d - 3 * (ph_d - pl_d)
    
    # Align to 12h timeframe
    r4_12h = align_htf_to_ltf(prices, df_1d, r4_d)
    s4_12h = align_htf_to_ltf(prices, df_1d, s4_d)
    
    # === Daily Trend Filter (EMA34) ===
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Volume Filter (2.0x 20-period EMA on 12h) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers daily calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_12h[i]) or np.isnan(s4_12h[i]) or 
            np.isnan(ema34_12h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price closes above R4 with uptrend and volume
            if (close[i] > r4_12h[i] and 
                close[i] > ema34_12h[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price closes below S4 with downtrend and volume
            elif (close[i] < s4_12h[i] and 
                  close[i] < ema34_12h[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below daily pivot (mean reversion)
            if close[i] < pp_d[i]:  # below daily pivot point
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price closes above daily pivot (mean reversion)
            if close[i] > pp_d[i]:  # above daily pivot point
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals
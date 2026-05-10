#!/usr/bin/env python3
# 4H_Pivot_Reversal_With_Volume_Spike
# Hypothesis: Mean reversion at daily Camarilla pivot levels (S1/R1) with volume spike confirmation.
# Long when: price touches S1 from below + volume > 2x average + price closes above S1.
# Short when: price touches R1 from above + volume > 2x average + price closes below R1.
# Exit when: price crosses the daily pivot point (PP) in opposite direction.
# Uses 1d Camarilla levels for structure, 4h for execution. Works in ranging markets (2025+).
# Target: 20-30 trades/year per symbol. Low frequency to avoid fee drag.

name = "4H_Pivot_Reversal_With_Volume_Spike"
timeframe = "4h"
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
    
    # Volume average (20-period)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels: S1, R1, PP
    # PP = (H + L + C) / 3
    # S1 = C - (H - L) * 1.1 / 12
    # R1 = C + (H - L) * 1.1 / 12
    pp = (prev_high + prev_low + prev_close) / 3
    s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    
    # Align daily levels to 4h timeframe (no look-ahead, uses previous day's close)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after volume MA warmup
        # Skip if any data not ready
        if (np.isnan(vol_ma[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 2.0
        
        if position == 0:
            # Long setup: price touches S1 from below with volume, closes above S1
            if (low[i] <= s1_aligned[i] and close[i] > s1_aligned[i] and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short setup: price touches R1 from above with volume, closes below R1
            elif (high[i] >= r1_aligned[i] and close[i] < r1_aligned[i] and volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below pivot point (mean reversion complete)
            if close[i] < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above pivot point
            if close[i] > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
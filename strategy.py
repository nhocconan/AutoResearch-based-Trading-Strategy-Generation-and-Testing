#!/usr/bin/env python3
# 6h_12h_Pivot_Reversal_Breakout
# Hypothesis: 6h price reversal at 12h pivot points (R2/S2) with volume confirmation.
# Enters long when price bounces off S2 with volume surge, short when rejected at R2.
# Uses 12h pivot points calculated from prior 12h bar's high/low/close.
# Designed for low trade frequency (15-25/year) to work in ranging and trending markets.

name = "6h_12h_Pivot_Reversal_Breakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get 12h data for pivot points
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h pivot points (classic: PP, R1, S1, R2, S2)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pivot point = (H + L + C) / 3
    pp = (high_12h + low_12h + close_12h) / 3.0
    # R2 = PP + (H - L)
    r2 = pp + (high_12h - low_12h)
    # S2 = PP - (H - L)
    s2 = pp - (high_12h - low_12h)
    
    # Align pivot levels to 6h timeframe
    r2_aligned = align_htf_to_ltf(prices, df_12h, r2)
    s2_aligned = align_htf_to_ltf(prices, df_12h, s2)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (2.0x average)
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: bounce off S2 with volume surge
            if low[i] <= s2_aligned[i] and close[i] > s2_aligned[i] and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: rejection at R2 with volume surge
            elif high[i] >= r2_aligned[i] and close[i] < r2_aligned[i] and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Long exit: price crosses below S2 or reverses at R2
                if close[i] < s2_aligned[i] or (high[i] >= r2_aligned[i] and close[i] < r2_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: price crosses above R2 or reverses at S2
                if close[i] > r2_aligned[i] or (low[i] <= s2_aligned[i] and close[i] > s2_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
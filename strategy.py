#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_Volume_TrendFilter
Hypothesis: Trade Camarilla Pivot R1/S1 breakouts on 4h with 1d trend filter (EMA34) and volume confirmation (>1.5x 24-bar average). Enter long when price breaks above R1 with bullish 1d EMA34 and volume surge; short when price breaks below S1 with bearish 1d EMA34 and volume surge. Exit when price returns to Pivot point or trend reverses. Designed for 20-35 trades/year via strict breakout conditions + trend alignment + volume filter. Works in bull/bear by following 1d trend. Uses Camarilla levels from prior 1d bar (no look-ahead).
"""

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
    
    # Get 1d data for Camarilla pivots and EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from prior 1d bar (H, L, C)
    # H = high, L = low, C = close of prior 1d bar
    H = df_1d['high'].values
    L = df_1d['low'].values
    C = df_1d['close'].values
    
    R1 = C + (H - L) * 1.1 / 12
    S1 = C - (H - L) * 1.1 / 12
    Pivot = (H + L + C) / 3
    
    # Align Camarilla levels to 4h (prior 1d bar known only after its close)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    Pivot_aligned = align_htf_to_ltf(prices, df_1d, Pivot)
    
    # 1d EMA(34) for trend filter
    if len(C) >= 34:
        ema_1d = np.full_like(C, np.nan)
        ema_1d[33] = np.mean(C[:34])
        for i in range(34, len(C)):
            ema_1d[i] = (C[i] * 2 / 35) + (ema_1d[i-1] * 33 / 35)
    else:
        ema_1d = np.full_like(C, np.nan)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 24:
        for i in range(24, len(volume)):
            vol_ma[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # volume MA needs 24 bars
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(Pivot_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price > R1 + volume + 1d EMA trending up (current > prior)
            if (close[i] > R1_aligned[i] and vol_confirm and 
                i > 0 and not np.isnan(ema_1d_aligned[i-1]) and ema_1d_aligned[i] > ema_1d_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: price < S1 + volume + 1d EMA trending down (current < prior)
            elif (close[i] < S1_aligned[i] and vol_confirm and 
                  i > 0 and not np.isnan(ema_1d_aligned[i-1]) and ema_1d_aligned[i] < ema_1d_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < Pivot or 1d EMA turns down
            if close[i] < Pivot_aligned[i] or (i > 0 and not np.isnan(ema_1d_aligned[i-1]) and ema_1d_aligned[i] < ema_1d_aligned[i-1]):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > Pivot or 1d EMA turns up
            if close[i] > Pivot_aligned[i] or (i > 0 and not np.isnan(ema_1d_aligned[i-1]) and ema_1d_aligned[i] > ema_1d_aligned[i-1]):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Volume_TrendFilter"
timeframe = "4h"
leverage = 1.0
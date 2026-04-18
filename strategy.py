#!/usr/bin/env python3
"""
4h_Pivot_R1S1_Breakout_1dTrend_Volume_Filtered
Hypothesis: Uses daily pivot point resistance/support levels with 1d EMA34 trend filter and volume confirmation.
Enters long when price breaks above R1 with EMA34 > EMA50 and volume spike, short when breaks below S1 with EMA34 < EMA50 and volume spike.
Designed for moderate trade frequency (~20-30/year) with strong trend capture in both bull and bear markets via pivot levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot points and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points (standard formula)
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    pivot = typical_price.values
    r1 = 2 * pivot - df_1d['low'].values
    s1 = 2 * pivot - df_1d['high'].values
    
    # Align pivot levels to 4h timeframe (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 1d EMA34 and EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = np.full(len(close_1d), np.nan)
    ema50_1d = np.full(len(close_1d), np.nan)
    
    # EMA34
    k34 = 2 / (34 + 1)
    for i in range(34, len(close_1d)):
        if i == 34:
            ema34_1d[i] = np.mean(close_1d[i-34:i+1])
        else:
            ema34_1d[i] = close_1d[i] * k34 + ema34_1d[i-1] * (1 - k34)
    
    # EMA50
    k50 = 2 / (50 + 1)
    for i in range(50, len(close_1d)):
        if i == 50:
            ema50_1d[i] = np.mean(close_1d[i-50:i+1])
        else:
            ema50_1d[i] = close_1d[i] * k50 + ema50_1d[i-1] * (1 - k50)
    
    # Align EMAs to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 with uptrend and volume spike
            if close[i] > r1_aligned[i] and ema34_1d_aligned[i] > ema50_1d_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with downtrend and volume spike
            elif close[i] < s1_aligned[i] and ema34_1d_aligned[i] < ema50_1d_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below S1 or trend weakens
            if close[i] < s1_aligned[i] or ema34_1d_aligned[i] <= ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above R1 or trend weakens
            if close[i] > r1_aligned[i] or ema34_1d_aligned[i] >= ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_R1S1_Breakout_1dTrend_Volume_Filtered"
timeframe = "4h"
leverage = 1.0
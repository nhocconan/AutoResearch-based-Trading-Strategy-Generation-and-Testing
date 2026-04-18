#!/usr/bin/env python3
"""
12h_1D_Pivot_R1S1_Breakout_Volume_Confirmation
Hypothesis: Uses daily pivot points (R1/S1) as key support/resistance levels on 12h chart.
Breakouts above R1 or below S1 with volume confirmation provide high-probability entries.
Trades in direction of 1w EMA trend filter to avoid counter-trend moves.
Designed for low frequency (15-25 trades/year) to minimize fee drag while capturing
major moves in both bull and bear markets via trend-aligned breakouts.
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
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points: (H + L + C) / 3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d      # Resistance 1
    s1 = 2 * pivot - high_1d     # Support 1
    
    # Get weekly data for trend filter (EMA 34)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        ema34_1w[33] = np.mean(close_1w[0:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1w)):
            ema34_1w[i] = close_1w[i] * alpha + ema34_1w[i-1] * (1 - alpha)
    
    # Volume spike: current volume > 1.5 x 24-period average (2 days)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    # Align daily pivot levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Align weekly EMA to 12h timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 34)
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 with volume spike and weekly uptrend
            if (close[i] > r1_aligned[i] and vol_spike[i] and 
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume spike and weekly downtrend
            elif (close[i] < s1_aligned[i] and vol_spike[i] and 
                  close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below S1 or weekly trend turns down
            if (close[i] < s1_aligned[i] or close[i] < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above R1 or weekly trend turns up
            if (close[i] > r1_aligned[i] or close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1D_Pivot_R1S1_Breakout_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0
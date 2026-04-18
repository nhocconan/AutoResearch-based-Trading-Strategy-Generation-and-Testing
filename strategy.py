#!/usr/bin/env python3
"""
1d_WeeklyPivot_TrendBreakout_1wEMA40_Volume
Hypothesis: Breakouts above weekly pivot R1 or below S1 levels on daily timeframe with volume spike and weekly EMA40 trend filter.
Weekly pivots provide key weekly support/resistance, volume confirms breakout strength, weekly trend filter avoids counter-trend trades.
Designed for low trade frequency (target: 7-25/year) with strong performance in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly EMA40 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA40 with proper smoothing
    ema40_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 40:
        ema40_1w[39] = np.mean(close_1w[0:40])
        alpha = 2 / (40 + 1)
        for i in range(40, len(close_1w)):
            ema40_1w[i] = close_1w[i] * alpha + ema40_1w[i-1] * (1 - alpha)
    
    # Align weekly EMA40 to daily timeframe
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    # Calculate weekly pivot levels (using previous week's high, low, close)
    prev_high = df_1w['high'].values
    prev_low = df_1w['low'].values
    prev_close = df_1w['close'].values
    
    # Calculate weekly pivot points and R1/S1 levels
    pivot = np.full(len(prev_close), np.nan)
    R1 = np.full(len(prev_close), np.nan)
    S1 = np.full(len(prev_close), np.nan)
    for i in range(len(prev_close)):
        if i == 0:  # First week has no previous week
            continue
        # Calculate using previous week's data
        ph = prev_high[i-1]
        pl = prev_low[i-1]
        pc = prev_close[i-1]
        pivot[i] = (ph + pl + pc) / 3.0
        range_val = ph - pl
        R1[i] = pivot[i] + range_val
        S1[i] = pivot[i] - range_val
    
    # Align weekly pivot levels to daily timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema40_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above weekly pivot R1 with volume spike and weekly uptrend
            if (close[i] > R1_aligned[i] and vol_spike[i] and 
                close[i] > ema40_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below weekly pivot S1 with volume spike and weekly downtrend
            elif (close[i] < S1_aligned[i] and vol_spike[i] and 
                  close[i] < ema40_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below weekly pivot S1 or weekly trend turns down
            if (close[i] < S1_aligned[i] or close[i] < ema40_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above weekly pivot R1 or weekly trend turns up
            if (close[i] > R1_aligned[i] or close[i] > ema40_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyPivot_TrendBreakout_1wEMA40_Volume"
timeframe = "1d"
leverage = 1.0
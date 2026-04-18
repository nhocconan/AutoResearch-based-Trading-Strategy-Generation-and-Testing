#!/usr/bin/env python3
"""
1d_WeeklyPivot_TrendBreakout_1wEMA40_Volume
Hypothesis: Daily breakouts above weekly pivot resistance (R2) or below support (S2) with weekly EMA40 trend filter and volume confirmation.
Weekly pivots capture key institutional levels, weekly EMA40 filters trend direction, volume confirms breakout strength.
Designed for low trade frequency (target: 10-25/year) with strong performance in both bull and bear markets via trend-following logic.
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
    
    # Get weekly data for pivot and EMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA40 trend filter
    ema40_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 40:
        ema40_1w[39] = np.mean(close_1w[0:40])
        alpha = 2 / (40 + 1)
        for i in range(40, len(close_1w)):
            ema40_1w[i] = close_1w[i] * alpha + ema40_1w[i-1] * (1 - alpha)
    
    # Calculate weekly pivot points (standard formula)
    pivot_1w = np.full(len(close_1w), np.nan)
    r1_1w = np.full(len(close_1w), np.nan)
    s1_1w = np.full(len(close_1w), np.nan)
    r2_1w = np.full(len(close_1w), np.nan)
    s2_1w = np.full(len(close_1w), np.nan)
    
    for i in range(1, len(close_1w)):
        # Use previous week's data for current week's pivot
        pp = (high_1w[i-1] + low_1w[i-1] + close_1w[i-1]) / 3.0
        r1 = 2 * pp - low_1w[i-1]
        s1 = 2 * pp - high_1w[i-1]
        r2 = pp + (high_1w[i-1] - low_1w[i-1])
        s2 = pp - (high_1w[i-1] - low_1w[i-1])
        
        pivot_1w[i] = pp
        r1_1w[i] = r1
        s1_1w[i] = s1
        r2_1w[i] = r2
        s2_1w[i] = s2
    
    # Align weekly indicators to daily timeframe
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Volume spike: current volume > 2.0 x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(r2_1w_aligned[i]) or np.isnan(s2_1w_aligned[i]) or 
            np.isnan(ema40_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above weekly R2 with volume spike and weekly uptrend
            if (close[i] > r2_1w_aligned[i] and vol_spike[i] and 
                close[i] > ema40_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S2 with volume spike and weekly downtrend
            elif (close[i] < s2_1w_aligned[i] and vol_spike[i] and 
                  close[i] < ema40_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below weekly S2 or weekly trend turns down
            if (close[i] < s2_1w_aligned[i] or close[i] < ema40_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above weekly R2 or weekly trend turns up
            if (close[i] > r2_1w_aligned[i] or close[i] > ema40_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyPivot_TrendBreakout_1wEMA40_Volume"
timeframe = "1d"
leverage = 1.0
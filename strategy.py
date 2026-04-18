#!/usr/bin/env python3
"""
1d_WeeklyPivot_TrendBreakout_1wEMA40_Volume
Hypothesis: Daily breakouts above weekly pivot resistance or below support with weekly EMA40 trend filter and volume confirmation.
Weekly pivots provide institutional reference points, weekly EMA40 filters trend direction, volume confirms breakout strength.
Designed for low trade frequency (target: 10-30/year) with strong performance in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # Calculate weekly pivot points (using previous week)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_arr = df_1w['close'].values
    
    # Calculate weekly pivot levels for each week
    pivot_1w = np.full(len(close_1w), np.nan)
    r1_1w = np.full(len(close_1w), np.nan)
    s1_1w = np.full(len(close_1w), np.nan)
    
    for i in range(1, len(close_1w)):
        pivot_1w[i] = (high_1w[i-1] + low_1w[i-1] + close_1w_arr[i-1]) / 3
        r1_1w[i] = 2 * pivot_1w[i] - low_1w[i-1]
        s1_1w[i] = 2 * pivot_1w[i] - high_1w[i-1]
    
    # Align weekly pivot levels to daily timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Volume spike: current volume > 1.8 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema40_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above weekly R1 with volume spike and weekly uptrend
            if (close[i] > r1_aligned[i] and vol_spike[i] and 
                close[i] > ema40_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S1 with volume spike and weekly downtrend
            elif (close[i] < s1_aligned[i] and vol_spike[i] and 
                  close[i] < ema40_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below weekly pivot or weekly trend turns down
            if (close[i] < pivot_aligned[i] or close[i] < ema40_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above weekly pivot or weekly trend turns up
            if (close[i] > pivot_aligned[i] or close[i] > ema40_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyPivot_TrendBreakout_1wEMA40_Volume"
timeframe = "1d"
leverage = 1.0
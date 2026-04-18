#!/usr/bin/env python3
"""
6h_WeeklyPivot_R2S2_Breakout_Volume_1wTrend
Hypothesis: Weekly pivot R2/S2 levels provide strong weekly support/resistance. Breakouts above R2 or below S2 with volume spike and weekly EMA20 trend filter capture major trend moves. Weekly trend filter avoids counter-trend trades. Designed for low trade frequency (target: 15-35/year) with strong performance in both bull and bear markets by trading with the weekly trend.
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
    
    # Calculate weekly EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA20 with proper smoothing
    ema20_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema20_1w[19] = np.mean(close_1w[0:20])
        alpha = 2 / (20 + 1)
        for i in range(20, len(close_1w)):
            ema20_1w[i] = close_1w[i] * alpha + ema20_1w[i-1] * (1 - alpha)
    
    # Align weekly EMA20 to 6h timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate weekly pivot points (using previous week's OHLC)
    prev_high = df_1w['high'].values
    prev_low = df_1w['low'].values
    prev_close = df_1w['close'].values
    
    # Calculate weekly pivot and R2/S2 levels
    pivot = np.full(len(prev_close), np.nan)
    R2 = np.full(len(prev_close), np.nan)
    S2 = np.full(len(prev_close), np.nan)
    for i in range(len(prev_close)):
        if i == 0:  # First week has no previous week
            continue
        # Calculate using previous week's data
        ph = prev_high[i-1]
        pl = prev_low[i-1]
        pc = prev_close[i-1]
        pivot[i] = (ph + pl + pc) / 3.0
        range_val = ph - pl
        R2[i] = pivot[i] + range_val
        S2[i] = pivot[i] - range_val
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    R2_aligned = align_htf_to_ltf(prices, df_1w, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1w, S2)
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(R2_aligned[i]) or np.isnan(S2_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above weekly R2 with volume spike and weekly uptrend
            if (close[i] > R2_aligned[i] and vol_spike[i] and 
                close[i] > ema20_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S2 with volume spike and weekly downtrend
            elif (close[i] < S2_aligned[i] and vol_spike[i] and 
                  close[i] < ema20_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below weekly pivot or weekly trend turns down
            if (close[i] < pivot_aligned[i] or close[i] < ema20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above weekly pivot or weekly trend turns up
            if (close[i] > pivot_aligned[i] or close[i] > ema20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R2S2_Breakout_Volume_1wTrend"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
"""
1d_WeeklyPivot_PivotBreakout_TrendFilter
Hypothesis: Weekly pivot breakouts on daily timeframe with volume confirmation and weekly trend filter. 
Weekly pivot levels provide strong support/resistance, volume confirms breakout momentum, 
weekly trend filter ensures alignment with higher timeframe bias. Designed for low frequency 
(10-30 trades/year) with strong performance in both bull and bear markets by avoiding 
counter-trend trades and capturing significant moves.
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
    
    # Get weekly data for pivot points and trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA40 trend filter
    close_1w = df_1w['close'].values
    ema40_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 40:
        ema40_1w[39] = np.mean(close_1w[0:40])
        alpha = 2 / (40 + 1)
        for i in range(40, len(close_1w)):
            ema40_1w[i] = close_1w[i] * alpha + ema40_1w[i-1] * (1 - alpha)
    
    # Calculate weekly pivot points using prior week's OHLC
    # Pivot = (H + L + C)/3, R1 = 2*P - L, S1 = 2*P - H
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = np.full(len(close_1w), np.nan)
    r1 = np.full(len(close_1w), np.nan)
    s1 = np.full(len(close_1w), np.nan)
    
    for i in range(len(close_1w)):
        if i == 0:  # First week has no previous week
            continue
        ph = high_1w[i-1]
        pl = low_1w[i-1]
        pc = close_1w[i-1]
        pivot[i] = (ph + pl + pc) / 3.0
        r1[i] = 2 * pivot[i] - pl
        s1[i] = 2 * pivot[i] - ph
    
    # Align weekly data to daily timeframe
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Volume spike: current volume > 2.0 x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
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

name = "1d_WeeklyPivot_PivotBreakout_TrendFilter"
timeframe = "1d"
leverage = 1.0
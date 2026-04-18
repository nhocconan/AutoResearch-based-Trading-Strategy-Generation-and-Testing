#!/usr/bin/env python3
"""
1d_WeeklyPivot_R1S1_Breakout_WeeklyTrend_Volume
Hypothesis: Weekly pivot R1/S1 breakout with weekly trend filter and volume confirmation.
Works in both bull and bear markets by trading breakouts from key weekly levels only when aligned
with weekly trend and accompanied by volume, avoiding false breakouts in ranging conditions.
Designed for low frequency (7-25 trades/year) with strong performance across market regimes.
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
    
    # Get weekly data for pivot levels and trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA34 trend filter
    close_1w = df_1w['close'].values
    ema34_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        ema34_1w[33] = np.mean(close_1w[0:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1w)):
            ema34_1w[i] = close_1w[i] * alpha + ema34_1w[i-1] * (1 - alpha)
    
    # Calculate weekly pivot points (based on prior week's OHLC)
    # Standard pivot: P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = np.full(len(close_1w), np.nan)
    r1 = np.full(len(close_1w), np.nan)  # R1 level
    s1 = np.full(len(close_1w), np.nan)  # S1 level
    
    for i in range(len(close_1w)):
        if i == 0:  # First week has no previous week
            continue
        ph = high_1w[i-1]
        pl = low_1w[i-1]
        pc = close_1w[i-1]
        pivot[i] = (ph + pl + pc) / 3.0
        r1[i] = 2 * pivot[i] - pl  # R1
        s1[i] = 2 * pivot[i] - ph  # S1
    
    # Align weekly data to daily timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above weekly R1 with volume spike and weekly uptrend
            if (close[i] > r1_aligned[i] and vol_spike[i] and 
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S1 with volume spike and weekly downtrend
            elif (close[i] < s1_aligned[i] and vol_spike[i] and 
                  close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below weekly S1 or weekly trend turns down
            if (close[i] < s1_aligned[i] or close[i] < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above weekly R1 or weekly trend turns up
            if (close[i] > r1_aligned[i] or close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyPivot_R1S1_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0
#!/usr/bin/env python3
"""
1d_WeeklyPivot_R1S1_Breakout_WeeklyTrend_Volume
Hypothesis: Weekly Pivot R1/S1 breakout with weekly EMA34 trend filter and daily volume confirmation.
Breakouts from key weekly levels only when aligned with weekly trend and accompanied by daily volume spike.
Designed for very low frequency (15-30 trades/year) to survive bear markets via tight entry conditions.
Works in bull via momentum continuation, in bear via mean-reversion from extreme weekly levels.
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
    
    # Get weekly data for EMA trend filter and pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA34 trend filter
    close_1w = df_1w['close'].values
    ema34_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        ema34_1w[33] = np.mean(close_1w[0:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1w)):
            ema34_1w[i] = close_1w[i] * alpha + ema34_1w[i-1] * (1 - alpha)
    
    # Calculate weekly Pivot points (R1/S1 from prior week)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    r1 = np.full(len(close_1w), np.nan)  # R1 level
    s1 = np.full(len(close_1w), np.nan)  # S1 level
    
    for i in range(1, len(close_1w)):
        ph = high_1w[i-1]
        pl = low_1w[i-1]
        pc = close_1w[i-1]
        pp = (ph + pl + pc) / 3.0
        r1[i] = pp + (ph - pl)  # R1
        s1[i] = pp - (ph - pl)  # S1
    
    # Align weekly EMA and pivot levels to daily timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Daily volume spike: current volume > 2.0 x 20-day average
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
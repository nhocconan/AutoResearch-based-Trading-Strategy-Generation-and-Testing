#!/usr/bin/env python3
"""
6h_Camarilla_R1S1_Breakout_WeeklyTrend_Volume
Hypothesis: Uses daily Camarilla pivot levels (R1/S1) for breakout entries with weekly trend filter and volume confirmation.
Long when price breaks above R1 with weekly close above weekly open and volume spike.
Short when price breaks below S1 with weekly close below weekly open and volume spike.
Designed for moderate trade frequency (~15-30/year) with strong trend capture in both bull and bear markets.
Weekly trend filter ensures alignment with higher timeframe momentum, reducing false breakouts in choppy markets.
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
    
    # Daily high, low, close for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    rng = high_1d - low_1d
    r1 = close_1d + rng * 1.1 / 12
    s1 = close_1d - rng * 1.1 / 12
    
    # Align to 6h timeframe (1 bar = 6h, 1d = 4 bars)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Weekly trend filter: weekly close > weekly open for uptrend, < for downtrend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_trend_up = weekly_close > weekly_open  # True for uptrend
    weekly_trend_down = weekly_close < weekly_open  # True for downtrend
    
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up)
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down)
    
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
            np.isnan(weekly_trend_up_aligned[i]) or np.isnan(weekly_trend_down_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 with weekly uptrend and volume spike
            if close[i] > r1_aligned[i] and weekly_trend_up_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with weekly downtrend and volume spike
            elif close[i] < s1_aligned[i] and weekly_trend_down_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below S1 or weekly trend turns down
            if close[i] < s1_aligned[i] or not weekly_trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above R1 or weekly trend turns up
            if close[i] > r1_aligned[i] or not weekly_trend_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R1S1_Breakout_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0
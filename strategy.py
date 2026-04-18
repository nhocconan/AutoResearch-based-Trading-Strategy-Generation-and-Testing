#!/usr/bin/env python3
"""
1d_Weekly_Camarilla_R1S1_Breakout_Volume
Hypothesis: Weekly timeframe is more stable, reducing noise. Price breaking through weekly R1/S1 with volume confirms institutional momentum. Using daily timeframe for entries with weekly context reduces whipsaw. Works in both bull (breakouts up) and bear (breakdowns down) by following price action. Volume confirmation avoids false breakouts. Target: 30-100 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for entries
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly close for trend filter
    weekly_close = df_1w['close'].values
    # Weekly EMA34 for trend
    weekly_ema34 = np.zeros_like(weekly_close)
    for i in range(len(weekly_close)):
        if i < 34:
            weekly_ema34[i] = np.mean(weekly_close[max(0, i-33):i+1]) if i >= 0 else weekly_close[i]
        else:
            weekly_ema34[i] = np.mean(weekly_close[i-33:i+1])
    
    # Align weekly EMA34 to daily
    weekly_ema34_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema34)
    
    # Calculate daily Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r1 = np.zeros_like(close_1d)
    camarilla_s1 = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i == 0:
            camarilla_r1[i] = close_1d[i]
            camarilla_s1[i] = close_1d[i]
        else:
            rang = high_1d[i-1] - low_1d[i-1]
            camarilla_r1[i] = close_1d[i-1] + rang * 1.1 / 12
            camarilla_s1[i] = close_1d[i-1] - rang * 1.1 / 12
    
    # Align to daily timeframe (use previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: current volume > 2.0x 20-day average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-20+1:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Warmup for weekly EMA34
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(weekly_ema34_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume spike AND weekly uptrend
            if (close[i] > camarilla_r1_aligned[i] and vol_spike[i] and 
                weekly_close[-1] > weekly_ema34_aligned[i]):  # weekly price above EMA34
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike AND weekly downtrend
            elif (close[i] < camarilla_s1_aligned[i] and vol_spike[i] and 
                  weekly_close[-1] < weekly_ema34_aligned[i]):  # weekly price below EMA34
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns below S1 or weekly trend changes
            if (close[i] < camarilla_s1_aligned[i] or 
                weekly_close[-1] < weekly_ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns above R1 or weekly trend changes
            if (close[i] > camarilla_r1_aligned[i] or 
                weekly_close[-1] > weekly_ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Camarilla_R1S1_Breakout_Volume"
timeframe = "1d"
leverage = 1.0
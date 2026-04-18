#!/usr/bin/env python3
"""
1d_Weekly_Camarilla_R1S1_Breakout_Volume
Hypothesis: Weekly Camarilla pivot levels act as strong support/resistance on the daily chart.
Price breaking through weekly R1/S1 with volume indicates institutional breakout.
Works in both bull and bear markets by following price action. Volume confirmation avoids false breakouts.
Daily timeframe keeps trades low (target 20-50/year) to minimize fee drag.
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
    
    # Get weekly data for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels from previous week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = np.zeros_like(close_1w)
    camarilla_s1 = np.zeros_like(close_1w)
    for i in range(len(close_1w)):
        if i == 0:
            camarilla_r1[i] = close_1w[i]  # placeholder
            camarilla_s1[i] = close_1w[i]
        else:
            rang = high_1w[i-1] - low_1w[i-1]
            camarilla_r1[i] = close_1w[i-1] + rang * 1.1 / 12
            camarilla_s1[i] = close_1w[i-1] - rang * 1.1 / 12
    
    # Align to daily timeframe (use previous week's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-20+1:i+1])
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for weekly alignment
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R1 with volume spike
            if close[i] > camarilla_r1_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with volume spike
            elif close[i] < camarilla_s1_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below weekly S1 (mean reversion) or volume dies
            if close[i] < camarilla_s1_aligned[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above weekly R1 or volume dies
            if close[i] > camarilla_r1_aligned[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Camarilla_R1S1_Breakout_Volume"
timeframe = "1d"
leverage = 1.0
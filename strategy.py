#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_HighLow_Strategy
Hypothesis: 12-hour breakouts above Camarilla R1 or below S1 with weekly EMA trend filter and volume confirmation.
Camarilla levels provide precise support/resistance from previous week, weekly EMA filters trend direction, volume confirms breakout strength.
Designed for low trade frequency (target: 15-30/year) with strong performance in both bull and bear markets.
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
    
    # Calculate weekly EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA34 with proper smoothing
    ema34_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        ema34_1w[33] = np.mean(close_1w[0:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1w)):
            ema34_1w[i] = close_1w[i] * alpha + ema34_1w[i-1] * (1 - alpha)
    
    # Align weekly EMA34 to 12h timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Camarilla levels from previous week using actual weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_arr = df_1w['close'].values
    
    # Calculate Camarilla levels for each week
    camarilla_r1_1w = np.full(len(close_1w), np.nan)
    camarilla_s1_1w = np.full(len(close_1w), np.nan)
    
    for i in range(1, len(close_1w)):
        range_val = high_1w[i-1] - low_1w[i-1]
        camarilla_r1_1w[i] = close_1w_arr[i-1] + range_val * 1.1 / 12
        camarilla_s1_1w[i] = close_1w_arr[i-1] - range_val * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1_1w)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1_1w)
    
    # Volume spike: current volume > 2.0 x 28-period average (for 12h: ~14 days)
    vol_ma = np.full(n, np.nan)
    for i in range(28, n):
        vol_ma[i] = np.mean(volume[i-28:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 28)
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above Camarilla R1 with volume spike and weekly uptrend
            if (close[i] > camarilla_r1_aligned[i] and vol_spike[i] and 
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Camarilla S1 with volume spike and weekly downtrend
            elif (close[i] < camarilla_s1_aligned[i] and vol_spike[i] and 
                  close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below Camarilla S1 or weekly trend turns down
            if (close[i] < camarilla_s1_aligned[i] or close[i] < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above Camarilla R1 or weekly trend turns up
            if (close[i] > camarilla_r1_aligned[i] or close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_HighLow_Strategy"
timeframe = "12h"
leverage = 1.0
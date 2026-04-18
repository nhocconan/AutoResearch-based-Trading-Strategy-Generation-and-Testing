#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrendFilter
Hypothesis: Uses 1d Camarilla pivot levels (R1/S1) on 12h timeframe with 1d EMA trend filter and volume confirmation. Designed for fewer trades (target 20-50/year) to work in both bull and bear markets by combining mean-reversion at pivots with trend filtering.
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
    
    # Get 1d data for Camarilla pivots and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
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
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = np.zeros_like(close_1d)
    if len(close_1d) >= 34:
        k = 2 / (34 + 1)
        ema_34_1d[33] = np.mean(close_1d[:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = close_1d[i] * k + ema_34_1d[i-1] * (1 - k)
    else:
        ema_34_1d[:] = np.nan
    
    # Align 1d indicators to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-20+1:i+1])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = 35  # Warmup for EMA
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: price breaks above R1 with volume spill and above 1d EMA34
            if close[i] > camarilla_r1_aligned[i] and vol_spike[i] and close[i] > ema_34_aligned[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: price breaks below S1 with volume spill and below 1d EMA34
            elif close[i] < camarilla_s1_aligned[i] and vol_spike[i] and close[i] < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Exit: minimum 6 bars hold, then exit on mean reversion or trend change
            if bars_since_entry >= 6:
                if close[i] < camarilla_s1_aligned[i] or close[i] < ema_34_aligned[i] or not vol_spike[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25  # Hold during minimum period
        
        elif position == -1:
            # Exit: minimum 6 bars hold, then exit on mean reversion or trend change
            if bars_since_entry >= 6:
                if close[i] > camarilla_r1_aligned[i] or close[i] > ema_34_aligned[i] or not vol_spike[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25  # Hold during minimum period
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrendFilter"
timeframe = "12h"
leverage = 1.0
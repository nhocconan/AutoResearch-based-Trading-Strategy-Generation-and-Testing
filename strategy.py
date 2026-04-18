#!/usr/bin/env python3
"""
4h_12h_1D_Camarilla_R1S1_Breakout_Volume_Trend
Hypothesis: Combines 12h Camarilla R1/S1 pivot breakouts with volume confirmation and 1d EMA34 trend filter.
Uses higher timeframe structure (12h) for signal direction and 1d trend filter to improve performance in both bull and bear markets.
Target: 25-40 trades/year. Uses Camarilla pivot levels which have proven effective for ETHUSDT and SOLUSDT.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 34:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivot levels
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Camarilla levels: R1, S1, R2, S2
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    R1 = np.full(len(high_12h), np.nan)
    S1 = np.full(len(high_12h), np.nan)
    R2 = np.full(len(high_12h), np.nan)
    S2 = np.full(len(high_12h), np.nan)
    
    for i in range(len(high_12h)):
        if i == 0:
            continue
        range_ = high_12h[i-1] - low_12h[i-1]
        if range_ <= 0:
            continue
        close_prev = close_12h[i-1]
        R1[i] = close_prev + range_ * 1.1 / 12
        S1[i] = close_prev - range_ * 1.1 / 12
        R2[i] = close_prev + range_ * 1.1 / 6
        S2[i] = close_prev - range_ * 1.1 / 6
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[0:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34_1d[i] = close_1d[i] * alpha + ema34_1d[i-1] * (1 - alpha)
    
    # Volume spike: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    # Align 12h Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_12h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_12h, S1)
    R2_aligned = align_htf_to_ltf(prices, df_12h, R2)
    S2_aligned = align_htf_to_ltf(prices, df_12h, S2)
    
    # Align 1d EMA34 to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Need EMA34 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 with volume spike and 1d uptrend
            if (close[i] > R1_aligned[i] and vol_spike[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume spike and 1d downtrend
            elif (close[i] < S1_aligned[i] and vol_spike[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below S1 or 1d trend turns down
            if (close[i] < S1_aligned[i] or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above R1 or 1d trend turns up
            if (close[i] > R1_aligned[i] or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_1D_Camarilla_R1S1_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0
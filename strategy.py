#!/usr/bin/env python3
# 6h_Camarilla_R1S1_Breakout_Volume_ATRFilter_v3
# Hypothesis: Camarilla pivot levels from 1d: long on breakout above R1 with volume spike,
# short on breakdown below S1 with volume spike. Filtered by 12h ATR(14) to avoid low volatility.
# Uses 12h ATR filter to ensure sufficient volatility for breakout moves.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_Camarilla_R1S1_Breakout_Volume_ATRFilter_v3"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    r1_1d = close_1d + 1.1 * (high_1d - low_1d) / 12
    s1_1d = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align pivots to 6h timeframe (use previous day's levels)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Get 12h data for ATR filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Calculate 12h ATR(14)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR calculation
    atr_12h = np.full_like(high_12h, np.nan)
    if len(high_12h) >= 14:
        atr_12h[14] = np.nanmean(tr_12h[1:15])
        for i in range(15, len(high_12h)):
            atr_12h[i] = (atr_12h[i-1] * 13 + tr_12h[i]) / 14
    
    # Align ATR to 6h timeframe
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Volume spike detection: volume > 1.5 * 20-period MA
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[20] = np.mean(volume[0:20])
        for i in range(21, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 15)  # volume MA and ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(atr_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR too small)
        if atr_12h_aligned[i] < 0.001 * close[i]:  # less than 0.1% of price
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: breakout above R1 with volume spike
            if close[i] > r1_1d_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below S1 with volume spike
            elif close[i] < s1_1d_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price returns below R1 or volatility drops
            if close[i] < r1_1d_aligned[i] or atr_12h_aligned[i] < 0.0005 * close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price returns above S1 or volatility drops
            if close[i] > s1_1d_aligned[i] or atr_12h_aligned[i] < 0.0005 * close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
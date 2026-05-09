#!/usr/bin/env python3
# 2025-06-22 | 1h_SMMA_Trend_Filter_v1
# Hypothesis: Use 4h Smoothed Moving Average (SMMA) for trend direction and 1h for precise entry timing.
# SMMA (Smoothed Moving Average) is less reactive than EMA/SMA, reducing whipsaws in sideways markets.
# Long when price > SMMA(50) and short when price < SMMA(50) on 4h timeframe.
# Entry on 1h only when price crosses SMMA with volume confirmation (>1.5x 20-period average).
# Designed for low trade frequency (15-35/year) to minimize fee drag in both bull and bear markets.

name = "1h_SMMA_Trend_Filter_v1"
timeframe = "1h"
leverage = 1.0

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
    
    # Get 4h data for SMMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h SMMA(50) - Smoothed Moving Average
    smma_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 50:
        # First value is simple average
        smma_4h[49] = np.mean(close_4h[0:50])
        # Subsequent values: SMMA = (PREV_SMMA * (N-1) + CLOSE) / N
        for i in range(50, len(close_4h)):
            smma_4h[i] = (smma_4h[i-1] * 49 + close_4h[i]) / 50
    
    # Align 4h SMMA to 1h timeframe
    smma_4h_aligned = align_htf_to_ltf(prices, df_4h, smma_4h)
    
    # Volume filter: 1h volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure SMMA and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(smma_4h_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price crosses above SMMA AND volume confirmation
            if close[i] > smma_4h_aligned[i] and volume_ratio[i] > 1.5:
                signals[i] = 0.20
                position = 1
            # Enter short: price crosses below SMMA AND volume confirmation
            elif close[i] < smma_4h_aligned[i] and volume_ratio[i] > 1.5:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below SMMA
            if close[i] < smma_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price crosses above SMMA
            if close[i] > smma_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_Regime
Hypothesis: TRIX momentum with volume spike confirmation and Choppiness regime filter.
Works in bull/bear by entering only when TRIX crosses zero with volume confirmation
and market is in trending regime (Choppiness < 38.2). Avoids false signals in ranging markets.
Designed for low frequency (15-30 trades/year) with strong performance across market regimes.
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
    
    # Get daily data for TRIX calculation and Choppiness filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate TRIX (15-period triple EMA) on daily close
    close_1d = df_1d['close'].values
    ema1 = np.full(len(close_1d), np.nan)
    ema2 = np.full(len(close_1d), np.nan)
    ema3 = np.full(len(close_1d), np.nan)
    
    if len(close_1d) >= 15:
        # First EMA
        alpha = 2 / (15 + 1)
        ema1[14] = np.mean(close_1d[0:15])
        for i in range(15, len(close_1d)):
            ema1[i] = close_1d[i] * alpha + ema1[i-1] * (1 - alpha)
        
        # Second EMA of EMA1
        ema2[29] = np.mean(ema1[15:30])  # Need 15 values of ema1
        for i in range(30, len(close_1d)):
            ema2[i] = ema1[i] * alpha + ema2[i-1] * (1 - alpha)
        
        # Third EMA of EMA2
        ema3[44] = np.mean(ema2[30:45])  # Need 15 values of ema2
        for i in range(45, len(close_1d)):
            ema3[i] = ema2[i] * alpha + ema3[i-1] * (1 - alpha)
    
    # TRIX = (EMA3 - previous EMA3) / previous EMA3 * 100
    trix = np.full(len(close_1d), np.nan)
    for i in range(46, len(close_1d)):
        if ema3[i-1] != 0:
            trix[i] = (ema3[i] - ema3[i-1]) / ema3[i-1] * 100
    
    # Calculate Choppiness Index (14-period) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    atr_1d = np.full(len(close_1d), np.nan)
    for i in range(1, len(close_1d)):
        tr = max(high_1d[i] - low_1d[i], 
                 abs(high_1d[i] - close_1d[i-1]), 
                 abs(low_1d[i] - close_1d[i-1]))
        if i == 1:
            atr_1d[i] = tr
        elif i < 14:
            atr_1d[i] = (atr_1d[i-1] * (i-1) + tr) / i
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr) / 14
    
    # Sum of true ranges over 14 periods
    tr_sum_14 = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        tr_sum_14[i] = np.sum(atr_1d[i-13:i+1])
    
    # Max(high-low) over 14 periods
    max_range_14 = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        max_range_14[i] = np.max(high_1d[i-13:i+1]) - np.min(low_1d[i-13:i+1])
    
    # Choppiness Index = 100 * log10(tr_sum_14 / max_range_14) / log10(14)
    chop = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        if max_range_14[i] > 0:
            chop[i] = 100 * np.log10(tr_sum_14[i] / max_range_14[i]) / np.log10(14)
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    # Align daily data to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(45, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(trix_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero with volume spike and trending regime (CHOP < 38.2)
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and 
                vol_spike[i] and chop_aligned[i] < 38.2):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with volume spike and trending regime (CHOP < 38.2)
            elif (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and 
                  vol_spike[i] and chop_aligned[i] < 38.2):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TRIX crosses below zero or chop indicates ranging (CHOP > 61.8)
            if (trix_aligned[i] < 0 or chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TRIX crosses above zero or chop indicates ranging (CHOP > 61.8)
            if (trix_aligned[i] > 0 or chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_TRIX_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0
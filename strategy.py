#!/usr/bin/env python3
"""
6h_12h_Camarilla_Pivot_Fade_Signal
Hypothesis: Fade price at Camarilla R3/S3 levels on 6h timeframe when 12h timeframe is in a ranging regime (Chop > 61.8). 
In ranging markets, price tends to revert from extreme levels. In trending markets (Chop < 38.2), we avoid trades to prevent whipsaw.
Works in bull/bear: Choppiness index regime filter ensures we only mean-revert in ranging conditions, avoiding trend-following losses.
Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.
"""

name = "6h_12h_Camarilla_Pivot_Fade_Signal"
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
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Choppiness Index for regime detection
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr = np.maximum(
        high_12h[1:] - low_12h[1:],
        np.maximum(
            np.abs(high_12h[1:] - close_12h[:-1]),
            np.abs(low_12h[1:] - close_12h[:-1])
        )
    )
    tr = np.concatenate([[high_12h[0] - low_12h[0]], tr])
    
    # ATR(14)
    atr = np.full(len(tr), np.nan)
    if len(tr) >= 14:
        atr[13] = np.mean(tr[:14])
        for i in range(14, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Chop = 100 * log10(sum(ATR14)/ (max(high)-min(low)) ) / log10(14)
    chop = np.full(len(close_12h), np.nan)
    for i in range(13, len(close_12h)):
        if atr[i] > 0:
            sum_atr14 = np.sum(atr[i-13:i+1])
            max_h = np.max(high_12h[i-13:i+1])
            min_l = np.min(low_12h[i-13:i+1])
            if max_h > min_l:
                chop[i] = 100 * np.log10(sum_atr14 / (max_h - min_l)) / np.log10(14)
            else:
                chop[i] = 50
        else:
            chop[i] = 50
    
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # Calculate Camarilla levels for 6h (using previous bar's OHLC)
    camarilla_high = np.full(n, np.nan)
    camarilla_low = np.full(n, np.nan)
    camarilla_close = np.full(n, np.nan)
    
    for i in range(1, n):
        camarilla_high[i] = high[i-1]
        camarilla_low[i] = low[i-1]
        camarilla_close[i] = close[i-1]
    
    # For first bar, use same values
    camarilla_high[0] = high[0]
    camarilla_low[0] = low[0]
    camarilla_close[0] = close[0]
    
    # Camarilla levels
    R3 = camarilla_close + (camarilla_high - camarilla_low) * 1.1000 / 4
    S3 = camarilla_close - (camarilla_high - camarilla_low) * 1.1000 / 4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 14  # Need Chop calculated
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in ranging market (Chop > 61.8)
        if chop_aligned[i] <= 61.8:
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Fade at S3 (support) - go long
            if close[i] <= S3[i]:
                signals[i] = 0.25
                position = 1
            # Fade at R3 (resistance) - go short
            elif close[i] >= R3[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price reaches midpoint or R3
            midpoint = camarilla_close[i] + (camarilla_high[i] - camarilla_low[i]) * 1.1000 / 2
            if close[i] >= midpoint or close[i] >= R3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price reaches midpoint or S3
            midpoint = camarilla_close[i] - (camarilla_high[i] - camarilla_low[i]) * 1.1000 / 2
            if close[i] <= midpoint or close[i] <= S3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_Volume_ChopRegime_V1
Hypothesis: 12h Camarilla R1/S1 breakouts with volume confirmation and 1d chop regime filter (CHOP > 61.8 = range) for mean reversion. 
In ranging markets (CHOP > 61.8), price tends to revert from extreme Camarilla levels (R1/S1). 
Volume confirmation (>1.5x 20-period MA) reduces false signals. 
Target 12-37 trades/year (50-150 total over 4 years) for 12h timeframe.
Uses 12h primary with 1d HTF for Camarilla calculation and chop regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla and chop regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Camarilla Pivot Levels (R1, S1) ===
    pivot = (high_1d + low_1d + close_1d) / 3.0
    rng = high_1d - low_1d
    r1 = pivot + (rng * 1.1 / 12)
    s1 = pivot - (rng * 1.1 / 12)
    
    # Align Camarilla levels (no extra delay needed - based on completed 1d bar)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 1d Choppiness Index (CHOP) for regime filter ===
    # CHOP = 100 * log10(sum(ATR(14)) / (n * (max(high) - min(low)))) / log10(n)
    # Simplified: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending
    tr1 = np.maximum(high_1d[1:] - low_1d[:-1], np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), np.abs(low_1d[1:] - close_1d[:-1])))
    tr1 = np.concatenate([[np.nan], tr1])  # align indices
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    sum_atr = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    max_min_range = max_high - min_low
    chop = 100 * np.log10(sum_atr / (14 * max_min_range)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 12h Indicators (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        vol = volume_12h[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        chop_ok = chop_aligned[i] > 61.8  # ranging regime (mean revert)
        
        if position == 0:
            # Long: price breaks below S1 + volume confirmation + chop regime (mean revert up)
            if price < s1_aligned[i] and vol_ok and chop_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks above R1 + volume confirmation + chop regime (mean revert down)
            elif price > r1_aligned[i] and vol_ok and chop_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price moves back above pivot (mean reversion complete) or chop breaks down
            if price > pivot[i] or chop_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price moves back below pivot (mean reversion complete) or chop breaks down
            if price < pivot[i] or chop_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_Volume_ChopRegime_V1"
timeframe = "12h"
leverage = 1.0
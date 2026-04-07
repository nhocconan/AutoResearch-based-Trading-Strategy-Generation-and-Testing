#!/usr/bin/env python3
"""
4h Camarilla Pivot + Volume Spike + Choppiness Regime
Long when price touches S1 with volume spike in choppy market
Short when price touches R1 with volume spike in choppy market
Exit when price moves to opposite pivot level or volume drops
Works in both bull/bear by fading extremes in ranging conditions
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_volume_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Daily Pivot Levels (using 1d data) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    s1 = close_1d - (range_1d * 1.12 / 12)
    r1 = close_1d + (range_1d * 1.12 / 12)
    s2 = close_1d - (range_1d * 1.12 / 6)
    r2 = close_1d + (range_1d * 1.12 / 6)
    
    # Align to 4h
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s2_4h = align_htf_to_ltf(prices, df_1d, s2)
    r2_4h = align_htf_to_ltf(prices, df_1d, r2)
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === Choppiness regime (using 1d data) ===
    # Chop = 100 * log10(sum(ATR)/ (highest_high - lowest_low)) / log10(n)
    atr_1d = []
    for i in range(len(df_1d)):
        if i == 0:
            tr = high_1d[i] - low_1d[i]
        else:
            tr = max(high_1d[i] - low_1d[i], 
                     abs(high_1d[i] - close_1d[i-1]),
                     abs(low_1d[i] - close_1d[i-1]))
        atr_1d.append(tr)
    
    chop = np.full(len(df_1d), 50.0)
    lookback = 14
    for i in range(lookback, len(df_1d)):
        sum_atr = sum(atr_1d[i-lookback+1:i+1])
        highest_high = np.max(high_1d[i-lookback+1:i+1])
        lowest_low = np.min(low_1d[i-lookback+1:i+1])
        if highest_high > lowest_low:
            chop[i] = 100 * np.log10(sum_atr) / np.log10(lookback) / np.log10((highest_high - lowest_low) + 1e-10)
        else:
            chop[i] = 50.0
    
    chop_4h = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        if np.isnan(pivot_4h[i]) or np.isnan(s1_4h[i]) or np.isnan(r1_4h[i]) or np.isnan(vol_ratio[i]) or np.isnan(chop_4h[i]):
            signals[i] = 0.0
            continue
        
        # Chop regime: trade only in choppy markets (61.8 > Chop > 38.2)
        if chop_4h[i] < 38.2 or chop_4h[i] > 61.8:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches R1 or volume drops
            if close[i] >= r1_4h[i] or vol_ratio[i] < 1.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches S1 or volume drops
            if close[i] <= s1_4h[i] or vol_ratio[i] < 1.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume spike
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: price near S1/R1 with volume
            if abs(close[i] - s1_4h[i]) / s1_4h[i] < 0.005:  # Within 0.5% of S1
                position = 1
                signals[i] = 0.25
            elif abs(close[i] - r1_4h[i]) / r1_4h[i] < 0.005:  # Within 0.5% of R1
                position = -1
                signals[i] = -0.25
    
    return signals
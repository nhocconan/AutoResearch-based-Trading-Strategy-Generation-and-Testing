#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + 1d Choppiness Filter.
Long when Alligator is bullish (jaw < teeth < lips) and chop < 61.8 (trending).
Short when Alligator is bearish (jaw > teeth > lips) and chop < 61.8 (trending).
Exit when Alligator alignment breaks or chop > 61.8 (range).
Uses 1d for chop regime, 12h for Alligator (SMMA-based).
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, period):
    """Smoothed Moving Average (SMMA) aka Wilder's MA"""
    if len(source) < period:
        return np.full_like(source, np.nan, dtype=float)
    sma = np.mean(source[:period])
    result = np.full_like(source, np.nan, dtype=float)
    result[period-1] = sma
    for i in range(period, len(source)):
        result[i] = (result[i-1] * (period-1) + source[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for chop regime
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d True Range and ATR(14) for chop denominator
    tr = np.maximum(high_1d - low_1d,
                    np.maximum(np.abs(high_1d - np.append([np.nan], close_1d[:-1])),
                               np.abs(low_1d - np.append([np.nan], close_1d[:-1]))))
    atr_14 = smma(tr, 14)
    
    # Calculate 1d Chopiness Index: 100 * log10(sum(ATR14) / (max(high)-min(low))) / log10(14)
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = np.maximum(max_high_14 - min_low_14, 1e-10)
    chop_raw = 100 * np.log10(sum_atr_14 / chop_denom) / np.log10(14)
    chop = np.where(np.isnan(chop_raw) | np.isinf(chop_raw), 50.0, chop_raw)  # neutral when invalid
    
    # Align 1d chop
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 12h Alligator (SMMA-based)
    # Jaw: SMMA(13, 8), Teeth: SMMA(8, 5), Lips: SMMA(5, 3)
    jaw = smma(close, 13)
    jaw = smma(jaw, 8)  # SMMA of SMMA
    teeth = smma(close, 8)
    teeth = smma(teeth, 5)
    lips = smma(close, 5)
    lips = smma(lips, 3)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i])):
            signals[i] = 0.0
            continue
        
        # Regime: trending when chop < 61.8
        is_trending = chop_aligned[i] < 61.8
        
        # Alligator alignment
        bullish = jaw[i] < teeth[i] and teeth[i] < lips[i]  # jaw < teeth < lips
        bearish = jaw[i] > teeth[i] and teeth[i] > lips[i]  # jaw > teeth > lips
        
        if position == 0:
            # Long: Alligator bullish AND trending
            if bullish and is_trending:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish AND trending
            elif bearish and is_trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator loses bullish alignment OR chop > 61.8 (range)
            if not bullish or chop_aligned[i] >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator loses bearish alignment OR chop > 61.8 (range)
            if not bearish or chop_aligned[i] >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dChop_Regime"
timeframe = "12h"
leverage = 1.0
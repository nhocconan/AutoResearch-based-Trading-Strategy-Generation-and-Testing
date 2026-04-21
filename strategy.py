#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_Volume_ChopRegime_ATRStop_V1
Hypothesis: 4h Camarilla R1/S1 breakouts with volume confirmation (>1.5x 20-period volume MA) and choppiness regime filter (CHOP > 50 for mean reversion). 
In ranging markets (CHOP > 50), price tends to revert from extreme Camarilla levels (R1/S1). 
Volume confirmation reduces false breakouts. ATR-based stoploss manages risk. 
Target 20-50 trades/year (80-200 total over 4 years) to avoid fee drag.
Uses 4h primary timeframe with 1d HTF for Camarilla calculation and 1h HTF for choppiness index.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla, 1h for choppiness)
    df_1d = get_htf_data(prices, '1d')
    df_1h = get_htf_data(prices, '1h')
    if len(df_1d) < 20 or len(df_1h) < 50:
        return np.zeros(n)
    
    # === 1d Camarilla Pivot Points (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1 = pivot + (range_1d * 1.1 / 12)
    s1 = pivot - (range_1d * 1.1 / 12)
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 1h Choppiness Index (CHOP) ===
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # True Range
    tr1 = np.maximum(high_1h[1:] - low_1h[1:], np.abs(high_1h[1:] - close_1h[:-1]))
    tr1 = np.maximum(tr1, np.abs(low_1h[1:] - close_1h[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])  # align with index
    
    atr_1h = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    # Chop = 100 * log10(sum(TR14) / (max(HH14) - min(LL14))) / log10(14)
    max_hh = pd.Series(high_1h).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low_1h).rolling(window=14, min_periods=14).min().values
    sum_tr = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    
    denominator = max_hh - min_ll
    chop_raw = 100 * np.log10(sum_tr / denominator) / np.log10(14)
    chop_raw = np.where(denominator == 0, 50, chop_raw)  # avoid division by zero
    chop_aligned = align_htf_to_ltf(prices, df_1h, chop_raw)
    
    # === 4h Indicators (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14) for stoploss
    tr_4h = np.maximum(high_4h[1:] - low_4h[1:], np.abs(high_4h[1:] - close_4h[:-1]))
    tr_4h = np.maximum(tr_4h, np.abs(low_4h[1:] - close_4h[:-1]))
    tr_4h = np.concatenate([[np.nan], tr_4h])
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop_multiplier = 2.5
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume_4h[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        chop = chop_aligned[i]
        in_chop_regime = chop > 50  # ranging market -> mean reversion
        
        if position == 0:
            # Long: price breaks below S1 (mean reversion long) + volume + chop regime
            if price < s1_aligned[i] and vol_ok and in_chop_regime:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks above R1 (mean reversion short) + volume + chop regime
            elif price > r1_aligned[i] and vol_ok and in_chop_regime:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Update signal
            signals[i] = 0.25
            # Exit conditions: price reverts back above pivot OR stoploss hit
            if price > pivot[i] or price < entry_price - atr_stop_multiplier * atr_4h[i]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Update signal
            signals[i] = -0.25
            # Exit conditions: price reverts back below pivot OR stoploss hit
            if price < pivot[i] or price > entry_price + atr_stop_multiplier * atr_4h[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Volume_ChopRegime_ATRStop_V1"
timeframe = "4h"
leverage = 1.0
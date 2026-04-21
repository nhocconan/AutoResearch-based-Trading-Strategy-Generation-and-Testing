#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_Volume_ChopRegime_ATRStop_V1
Hypothesis: 4h Camarilla pivot breakouts at R1/S1 with volume confirmation (>1.3x 20-period volume MA) and choppiness regime filter (CHOP < 61.8 = trending market). 
Camarilla levels from 1d HTF provide institutional support/resistance. Volume confirms participation. 
Choppiness filter avoids whipsaws in ranging markets. Target 20-50 trades/year (80-200 total over 4 years).
Uses 4h primary timeframe with 1d HTF for Camarilla calculation and chop filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla pivots and chop filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # === 1d Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    camarilla_r1 = close_1d + (range_1d * 1.1 / 12.0)
    camarilla_s1 = close_1d - (range_1d * 1.1 / 12.0)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === 1d Choppiness Index (CHOP) for regime filter ===
    # CHOP = 100 * log10(sum(ATR(14)) / (n * (max(high) - min(low)))) / log10(n)
    # Simplified: CHOP < 38.2 = strong trend, CHOP > 61.8 = ranging/choppy
    # We use CHOP < 61.8 to allow trending markets only
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    # Pad first element
    tr_1d = np.concatenate([[np.abs(high_1d[0] - low_1d[0])], tr_1d])
    
    atr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Avoid division by zero
    chop_1d = np.where(
        range_14 > 0,
        100 * np.log10(sum_atr_14 / (14 * range_14)) / np.log10(14),
        50.0  # neutral when range is zero
    )
    
    # Align chop to 4h timeframe (trending market: CHOP < 61.8)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 4h Indicators (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Volume MA (20-period) for confirmation
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) 
            or np.isnan(chop_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume_4h[i]
        vol_ok = vol > 1.3 * vol_ma[i]  # volume confirmation
        chop_ok = chop_1d_aligned[i] < 61.8  # trending market regime
        
        if position == 0:
            # Long: price breaks above R1 + volume confirmation + trending market
            if price > camarilla_r1_aligned[i] and vol_ok and chop_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume confirmation + trending market
            elif price < camarilla_s1_aligned[i] and vol_ok and chop_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S1 or chop becomes too high (ranging)
            if price < camarilla_s1_aligned[i] or chop_1d_aligned[i] >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 or chop becomes too high (ranging)
            if price > camarilla_r1_aligned[i] or chop_1d_aligned[i] >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Volume_ChopRegime_ATRStop_V1"
timeframe = "4h"
leverage = 1.0
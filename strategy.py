#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_Volume_ChopRegime_V1
Hypothesis: 12h Camarilla R1/S1 breakout with volume confirmation (>1.5x 20-period volume MA) and 1d choppiness regime filter (CHOP > 61.8 = range = mean reversion, CHOP < 38.2 = trending = follow breakout direction). 
In ranging markets (CHOP > 61.8), we fade the breakout (short at R1, long at S1). 
In trending markets (CHOP < 38.2), we follow the breakout (long at R1 break, short at S1 break).
Uses 12h primary timeframe with 1d HTF for Camarilla calculation and chop filter.
Target 12-37 trades/year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla and chop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d Camarilla Pivot Levels (R1, S1, R4, S4, PP) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point
    pp = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r1 = pp + (range_1d * 1.1 / 12)
    s1 = pp - (range_1d * 1.1 / 12)
    r4 = pp + (range_1d * 1.1 / 2)
    s4 = pp - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 1d Choppiness Index (CHOP) ===
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    # Set first TR to high-low (no previous close)
    tr[0] = tr1[0]
    
    # ATR(14) - sum of TR over 14 periods
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Max/min close over 14 periods
    max_close_14 = pd.Series(close_1d).rolling(window=14, min_periods=14).max().values
    min_close_14 = pd.Series(close_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(ATR(14) / (max_close - min_close)) / log10(14)
    # Avoid division by zero
    range_14 = max_close_14 - min_close_14
    chop = np.zeros_like(close_1d)
    mask = (range_14 > 0) & (atr_14 > 0)
    chop[mask] = 100 * np.log10(atr_14[mask] / range_14[mask]) / np.log10(14)
    # For invalid cases, set to 50 (neutral)
    chop[~mask] = 50.0
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 12h Indicators (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
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
        chop_val = chop_aligned[i]
        
        if position == 0:
            # Determine market regime
            is_ranging = chop_val > 61.8  # CHOP > 61.8 = ranging (mean reversion)
            is_trending = chop_val < 38.2  # CHOP < 38.2 = trending (follow breakout)
            
            if is_ranging:
                # In ranging markets: fade the breakout
                # Short at R1 break, Long at S1 break
                if price > r1_aligned[i] and vol_ok:
                    signals[i] = -0.25  # short at R1 break
                    position = -1
                elif price < s1_aligned[i] and vol_ok:
                    signals[i] = 0.25   # long at S1 break
                    position = 1
            elif is_trending:
                # In trending markets: follow the breakout
                # Long at R1 break, Short at S1 break
                if price > r1_aligned[i] and vol_ok:
                    signals[i] = 0.25   # long at R1 break
                    position = 1
                elif price < s1_aligned[i] and vol_ok:
                    signals[i] = -0.25  # short at S1 break
                    position = -1
            # In neutral zone (38.2 <= CHOP <= 61.8): no action
        
        elif position == 1:
            # Exit long: price reaches S4 (strong support) or reverses below S1
            if price < s4_aligned[i] or price < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches R4 (strong resistance) or reverses above R1
            if price > r4_aligned[i] or price > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_Volume_ChopRegime_V1"
timeframe = "12h"
leverage = 1.0
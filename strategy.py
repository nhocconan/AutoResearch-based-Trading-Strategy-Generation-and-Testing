#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_Chop_ATRStop
Hypothesis: 4h Camarilla R1/S1 breakouts with volume confirmation and chop regime filter provide edge in both bull and bear markets. 
Long when price breaks above R1 with volume > 1.5x average and chop < 61.8 (trending). 
Short when price breaks below S1 with volume > 1.5x average and chop < 61.8 (trending). 
Uses ATR-based stoploss via signal=0 when price moves against position by 2.0*ATR.
Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.
Works in trending markets (chop < 61.8) and avoids ranging markets where breakouts fail.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla calculation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Camarilla levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_range = high_1d - low_1d
    r1 = close_1d + (1.1 * camarilla_range / 12)
    s1 = close_1d - (1.1 * camarilla_range / 12)
    
    # Align to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 4h Indicators (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Volume average (20-period) for confirmation
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR(1)) / (max(high) - min(low))) / log10(N)
    tr1 = pd.Series(high_4h - low_4h)
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1 = tr.rolling(window=1, min_periods=1).sum()
    sum_atr = atr_1.rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    chop = 100 * (np.log10(sum_atr) - np.log10(max_high - min_low)) / np.log10(14)
    
    # ATR (14-period) for stoploss
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(chop[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        volume = volume_4h[i]
        
        if position == 0:
            # Long: Price breaks above R1 + volume > 1.5x average + trending regime (CHOP < 61.8)
            if price > r1_aligned[i] and volume > 1.5 * vol_ma[i] and chop[i] < 61.8:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Price breaks below S1 + volume > 1.5x average + trending regime (CHOP < 61.8)
            elif price < s1_aligned[i] and volume > 1.5 * vol_ma[i] and chop[i] < 61.8:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: price back below R1 or chop increases (ranging market)
            elif price < r1_aligned[i] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: price back above S1 or chop increases (ranging market)
            elif price > s1_aligned[i] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_Chop_ATRStop"
timeframe = "4h"
leverage = 1.0
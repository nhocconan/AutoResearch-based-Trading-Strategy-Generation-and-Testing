#!/usr/bin/env python3
"""
6h_Camarilla_R4_S4_Breakout_VolumeFilter
Hypothesis: 6h Camarilla pivot R4/S4 breakout with volume confirmation (>1.8x 20-period volume MA) and 1d HTF trend filter (price > EMA50 for longs, < EMA50 for shorts). R4/S4 levels represent stronger breakout points than R1/S1, reducing false signals and trade frequency. Designed for low trade frequency (<100 total 6h trades over 4 years) to minimize fee drag and capture strong momentum moves in both bull and bear markets via trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h Indicators (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate Camarilla pivot levels from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r4 = pivot + (range_1d * 1.1 / 2)   # R4: strong breakout level
    s4 = pivot - (range_1d * 1.1 / 2)   # S4: strong breakdown level
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        vol = volume_6h[i]
        vol_ok = vol > 1.8 * vol_ma[i]  # volume confirmation
        
        if position == 0:
            # Long: Camarilla S4 breakdown (wait for retest) + volume + uptrend
            # Actually, we want breakout ABOVE R4 for long, BELOW S4 for short
            if price > r4_aligned[i] and vol_ok and price > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif price < s4_aligned[i] and vol_ok and price < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit: price retests below R4 or loss of volume/uptrend
            if price < r4_aligned[i] or not vol_ok or price < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price retests above S4 or loss of volume/downtrend
            if price > s4_aligned[i] or not vol_ok or price > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R4_S4_Breakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_Volume_ATRFilter_V1
Hypothesis: 12h Camarilla pivot R1/S1 breakouts with volume confirmation (>1.5x 20-period volume MA) and ATR-based stoploss. 
Camarilla levels derived from 1d HTF provide institutional support/resistance. 
Volume confirmation reduces false breakouts. ATR stoploss manages risk. 
Target 12-37 trades/year (50-150 total over 4 years).
Uses 12h primary timeframe with 1d HTF for Camarilla calculation and 1w HTF for trend regime (price > 1w EMA50 for longs, < for shorts).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla, 1w for trend regime)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # === 1d Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical Price for pivot calculation
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla width based on 1d range
    camarilla_width = (high_1d - low_1d) * 1.1 / 12.0
    # Camarilla R1 and S1 levels
    r1 = close_1d + camarilla_width * 1.1
    s1 = close_1d - camarilla_width * 1.1
    
    # Align Camarilla levels to 12h timeframe (no extra delay needed for pivot points)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 1w EMA50 for trend regime filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 12h Indicators (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss
    tr1 = np.maximum(high_12h[1:] - low_12h[1:], np.abs(high_12h[1:] - close_12h[:-1]))
    tr2 = np.maximum(np.abs(low_12h[1:] - close_12h[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])  # first tr is NaN
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close_12h[i]
        vol = volume_12h[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        if position == 0:
            # Long: price breaks above R1 + volume confirmation + 1w uptrend
            if price > r1_aligned[i] and vol_ok and price > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S1 + volume confirmation + 1w downtrend
            elif price < s1_aligned[i] and vol_ok and price < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long position management
            # Stoploss: 2 * ATR below entry
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below 1w EMA50 or volume confirmation fails
            elif price < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short position management
            # Stoploss: 2 * ATR above entry
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above 1w EMA50 or volume confirmation fails
            elif price > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_Volume_ATRFilter_V1"
timeframe = "12h"
leverage = 1.0
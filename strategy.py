#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_VolumeATRFilter_V1
Hypothesis: Use 1d Camarilla R1/S1 levels from prior day + 12h breakout with volume spike (>1.5x 20-bar MA) and ATR(14) stoploss (1.5x). 12h timeframe reduces noise, Camarilla levels provide institutional support/resistance, volume confirms legitimacy, ATR stop manages risk. Designed for ranging/low-volatility markets like 2025 BTC/ETH.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')  # for Camarilla levels
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d Camarilla Levels (from prior day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for today using yesterday's OHLC
    camarilla_r1 = np.zeros(len(close_1d))
    camarilla_s1 = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        # Yesterday's values
        h = high_1d[i-1]
        l = low_1d[i-1]
        c = close_1d[i-1]
        camarilla_r1[i] = c + (h - l) * 1.1 / 12
        camarilla_s1[i] = c - (h - l) * 1.1 / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === 12h Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume spike confirmation
        
        if position == 0:
            # Long: break above Camarilla R1 with volume spike
            if price > camarilla_r1_aligned[i-1] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: break below Camarilla S1 with volume spike
            elif price < camarilla_s1_aligned[i-1] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: ATR stoploss or opposite signal
            if price < camarilla_r1_aligned[i-1] - 1.5 * atr[i] or (price < camarilla_s1_aligned[i-1] and vol_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: ATR stoploss or opposite signal
            if price > camarilla_s1_aligned[i-1] + 1.5 * atr[i] or (price > camarilla_r1_aligned[i-1] and vol_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_VolumeATRFilter_V1"
timeframe = "12h"
leverage = 1.0
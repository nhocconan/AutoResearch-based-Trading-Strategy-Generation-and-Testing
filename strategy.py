#!/usr/bin/env python3
"""
6h_1d_Camarilla_R1S1_Breakout_Volume_Control_v1
Hypothesis: Breakout of daily Camarilla R1/S1 levels with volume confirmation and ATR filter on 6h timeframe.
Long when price > daily R1 + volume spike + ATR(14) > 0.02 * price.
Short when price < daily S1 + volume spike + ATR(14) > 0.02 * price.
Exit when price crosses daily pivot point (PP).
Uses 6h as primary timeframe for signal generation, with 1d for levels.
Designed to work in both bull and bear markets by requiring volatility and volume confirmation.
Target: 12-37 trades per year per symbol (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: R1, S1, and pivot point (PP)
    rang = prev_high - prev_low
    r1 = prev_close + 1.1 * rang / 12
    s1 = prev_close - 1.1 * rang / 12
    pp = (prev_high + prev_low + prev_close) / 3
    
    # Align to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Calculate ATR(14) on 6h data for volatility filter
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr = np.zeros_like(tr)
    for i in range(len(tr)):
        if i < 14:
            atr[i] = np.nan
        else:
            atr[i] = np.mean(tr[i-13:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # ATR filter: volatility > 2% of price
        atr_ok = atr[i] > 0.02 * price
        
        if position == 0:
            # Long conditions: break above R1 + volume + ATR
            if price > r1_aligned[i] and volume_ok and atr_ok:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below S1 + volume + ATR
            elif price < s1_aligned[i] and volume_ok and atr_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below pivot point
            if price < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above pivot point
            if price > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_Camarilla_R1S1_Breakout_Volume_Control_v1"
timeframe = "6h"
leverage = 1.0
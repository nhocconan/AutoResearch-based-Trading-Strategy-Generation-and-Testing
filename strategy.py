#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_VolumeATRFilter_V1
Hypothesis: Camarilla pivot R1/S1 breakout with volume confirmation (>1.5x 20-bar MA) and ATR-based stoploss works on 12h timeframe for BTC and ETH in both bull and bear markets. Uses 1d timeframe for pivot calculation. Target: 12-30 trades/year per symbol (50-120 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla pivot calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot points (R1, S1) from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align Camarilla levels to 12h timeframe (completed 1d bar only)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume filter: 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation (>1.5x average)
        volume_ok = volume > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume
            if price > r1_aligned[i] and volume_ok:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S1 with volume
            elif price < s1_aligned[i] and volume_ok:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long: exit on stoploss or reverse signal
            if price < entry_price - 2.0 * atr[i] or price < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short: exit on stoploss or reverse signal
            if price > entry_price + 2.0 * atr[i] or price > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_VolumeATRFilter_V1"
timeframe = "12h"
leverage = 1.0
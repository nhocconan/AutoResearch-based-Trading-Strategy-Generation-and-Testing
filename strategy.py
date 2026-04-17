#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_v1
Based on proven pattern: Camarilla pivot levels from 1d + volume spike + ATR stoploss.
Long when price breaks above R1 with volume confirmation and ATR-based volatility filter.
Short when price breaks below S1 with volume confirmation and ATR-based volatility filter.
Exit when price returns to the Pivot point or ATR volatility drops.
Target: 50-150 total trades over 4 years (12-37/year).
Works in both bull and bear markets due to mean-reversion at extreme levels with volatility filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1d OHLC for Camarilla pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    o_1d = df_1d['open'].values
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_1d = (h_1d + l_1d + c_1d) / 3.0
    r1_1d = c_1d + (h_1d - l_1d) * 1.1 / 12.0
    s1_1d = c_1d - (h_1d - l_1d) * 1.1 / 12.0
    
    # Align to 12h timeframe (wait for 1d bar to close)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === Volume spike detection (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)  # Avoid division by zero
    
    # === ATR for volatility filter and stoploss ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(vol_ratio[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above R1 with volume spike (vol_ratio > 1.5) and ATR > 0
            if (close[i] > r1_1d_aligned[i] and 
                vol_ratio[i] > 1.5 and 
                atr[i] > 0):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S1 with volume spike (vol_ratio > 1.5) and ATR > 0
            elif (close[i] < s1_1d_aligned[i] and 
                  vol_ratio[i] > 1.5 and 
                  atr[i] > 0):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price returns to pivot point OR ATR drops significantly (volatility collapse)
            if (close[i] <= pivot_1d_aligned[i] or 
                atr[i] < 0.5 * atr[max(0, i-1)]):  # ATR dropped more than 50% from previous bar
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to pivot point OR ATR drops significantly
            if (close[i] >= pivot_1d_aligned[i] or 
                atr[i] < 0.5 * atr[max(0, i-1)]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_v1"
timeframe = "12h"
leverage = 1.0
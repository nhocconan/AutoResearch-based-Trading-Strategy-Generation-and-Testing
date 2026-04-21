#!/usr/bin/env python3
"""
4h_1d_HTF_Camarilla_Pivot_VolumeBreakout_V1
Hypothesis: Trade Camarilla pivot breakouts (R1/S1) from 1d timeframe on 4h chart with volume confirmation (>1.5x 20-bar MA) and ATR(14) stoploss (1.5x). Uses 1d HTF for pivot levels to reduce noise and focus on significant intraday breaks. Works in bull/bear by capturing meaningful breakouts with volume validation while avoiding choppy markets via ATR-based stops. Target 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')  # for Camarilla pivot levels
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d Camarilla Pivot Levels (based on previous day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and levels from previous 1d bar
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1_1d = pivot_1d + (range_1d * 1.1 / 12)
    s1_1d = pivot_1d - (range_1d * 1.1 / 12)
    
    # Align to 4h timeframe (values available after 1d bar closes)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 4h Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume MA (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss
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
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i])
            or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        if position == 0:
            # Long: break above 1d R1 with volume
            if price > r1_1d_aligned[i-1] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: break below 1d S1 with volume
            elif price < s1_1d_aligned[i-1] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: ATR stoploss or opposite break
            if price < r1_1d_aligned[i-1] - 1.5 * atr[i] or price < s1_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: ATR stoploss or opposite break
            if price > s1_1d_aligned[i-1] + 1.5 * atr[i] or price > r1_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_HTF_Camarilla_Pivot_VolumeBreakout_V1"
timeframe = "4h"
leverage = 1.0
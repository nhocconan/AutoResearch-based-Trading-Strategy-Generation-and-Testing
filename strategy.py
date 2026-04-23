#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 breakout with 1d ADX trend filter and volume confirmation.
Long when price breaks above 1d Camarilla R1 AND 1d ADX > 25 AND volume > 1.5x 20-period average.
Short when price breaks below 1d Camarilla S1 AND 1d ADX > 25 AND volume > 1.5x 20-period average.
Exit when price touches the opposite Camarilla level (S1 for longs, R1 for shorts).
Uses 1d HTF for Camarilla levels and ADX trend strength (avoids whipsaws in ranging markets).
Target: 50-150 total trades over 4 years (12-37/year).
Camarilla pivot breakouts capture institutional order flow; ADX filter ensures we only trade in trending regimes.
Works in both bull and bear markets when trends exist (ADX > 25).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d HTF data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (R1, S1)
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_r1_1d = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1_1d = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Calculate 1d ADX for trend filter
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate ADX components
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d HTF indicators to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14 + 13 + 13)  # volume MA (20), ADX calculation (14+13+13)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        adx_val = adx_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Break above Camarilla R1 AND ADX > 25 AND volume spike
            if price > r1 and adx_val > 25 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S1 AND ADX > 25 AND volume spike
            elif price < s1 and adx_val > 25 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit when price touches opposite Camarilla level
            if position == 1 and price < s1:  # Long exit at Camarilla S1
                exit_signal = True
            elif position == -1 and price > r1:  # Short exit at Camarilla R1
                exit_signal = True
            else:
                exit_signal = False
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R1S1_Breakout_1dADX25_Trend_VolumeConfirmation_LevelExit"
timeframe = "12h"
leverage = 1.0
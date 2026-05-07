#!/usr/bin/env python3
name = "6h_ADX_Alligator_Trend"
timeframe = "6h"
leverage = 1.0

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
    
    # Load daily data ONCE before loop for ADX and Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    
    # Alligator (Jaw: 13, Teeth: 8, Lips: 5) - all shifted
    jaw = pd.Series(close_1d).rolling(window=13, center=False).mean().shift(8).values
    teeth = pd.Series(close_1d).rolling(window=8, center=False).mean().shift(5).values
    lips = pd.Series(close_1d).rolling(window=5, center=False).mean().shift(3).values
    
    # Align ADX and Alligator components to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 14  # Wait for ADX calculation
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Strong trend and Alligator aligned for long
            if adx_aligned[i] > 25 and lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Strong trend and Alligator aligned for short
            elif adx_aligned[i] > 25 and lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: trend weakens or Alligator reverses
            if adx_aligned[i] < 20 or lips_aligned[i] < teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: trend weakens or Alligator reverses
            if adx_aligned[i] < 20 or lips_aligned[i] > teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6s ADX + Alligator trend following with strict alignment filters
# - Uses daily ADX(14) to identify strong trends (>25) and avoid ranging markets (<20)
# - Uses daily Alligator (Jaw/Teeth/Lips) to confirm trend direction and alignment
# - Long when ADX>25 and Lips>Teeth>Jaw (bullish alignment)
# - Short when ADX>25 and Lips<Teeth<Jaw (bearish alignment)
# - Exits when ADX<20 (trend weakening) or Alligator misaligns
# - Works in both bull and bear markets by capturing strong trends in either direction
# - Avoids whipsaws in ranging markets via ADX filter
# - Position size 0.25 limits risk while capturing trend moves
# - Targets 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# - Novel combination: ADX trend strength + Alligator alignment not recently tried on 6h
# - Uses proper smoothing with ewm and rolling windows for accurate indicator values
#!/usr/bin/env python3
"""
4h_1d_1w_TRIX_VolumeSpike_Momentum_v1
Concept: TRIX momentum with volume spike confirmation and trend filter from weekly EMA.
- TRIX(12) crossing zero line with volume > 1.8x 20-period average for entry
- Long: TRIX crosses above zero, volume spike, close > weekly EMA200
- Short: TRIX crosses below zero, volume spike, close < weekly EMA200
- Exit: TRIX crosses back through zero in opposite direction
- Position sizing: 0.28 (balanced for trend capture and drawdown control)
- Target: ~80-120 trades over 4 years to minimize fee drag
- Works in bull/bear: TRIX catches momentum shifts, volume filter avoids false signals, weekly EMA ensures trend alignment
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_TRIX_VolumeSpike_Momentum_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 20:
        return np.zeros(n)
    
    # === 1d: TRIX Calculation ===
    close_1d = df_1d['close'].values
    # EMA1
    ema1 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA2
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA3
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    # TRIX = (EMA3 - prev EMA3) / prev EMA3 * 100
    trix = np.zeros_like(close_1d)
    trix[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    trix[0] = 0
    
    # === 1w: EMA200 for trend filter ===
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align indicators to 4h
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # === 4h: Indicators ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume: 20-period average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Get values
        trix_val = trix_aligned[i]
        trix_prev = trix_aligned[i-1] if i > 0 else 0
        ema200_val = ema200_1w_aligned[i]
        vol_ma_val = vol_ma[i]
        current_volume = volume[i]
        current_close = close[i]
        
        # Skip if any value is NaN
        if (np.isnan(trix_val) or np.isnan(trix_prev) or np.isnan(ema200_val) or 
            np.isnan(vol_ma_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.8x 20-period average
        vol_condition = current_volume > 1.8 * vol_ma_val
        
        # TRIX zero cross detection
        trix_cross_up = trix_prev <= 0 and trix_val > 0
        trix_cross_down = trix_prev >= 0 and trix_val < 0
        
        if position == 0:
            # Long: TRIX crosses up through zero with volume and above weekly EMA200
            if trix_cross_up and vol_condition and current_close > ema200_val:
                signals[i] = 0.28
                position = 1
            # Short: TRIX crosses down through zero with volume and below weekly EMA200
            elif trix_cross_down and vol_condition and current_close < ema200_val:
                signals[i] = -0.28
                position = -1
        
        elif position == 1:
            # Long exit: TRIX crosses back down through zero
            if trix_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        
        elif position == -1:
            # Short exit: TRIX crosses back up through zero
            if trix_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals
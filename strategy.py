#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_R4S4_Breakout_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # === Weekly Pivot Points (Standard) ===
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot_w = (high_w + low_w + close_w) / 3.0
    # R4 = 3*P - 2*L, S4 = 3*P - 2*H
    r4_w = 3 * pivot_w - 2 * low_w
    s4_w = 3 * pivot_w - 2 * high_w
    
    # Align weekly levels to 6h
    pivot_w_aligned = align_htf_to_ltf(prices, df_1w, pivot_w)
    r4_w_aligned = align_htf_to_ltf(prices, df_1w, r4_w)
    s4_w_aligned = align_htf_to_ltf(prices, df_1w, s4_w)
    
    # === 6h: Volume and ATR for filtering ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # TR for ATR
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # Enough for ATR and volume MA
    
    for i in range(start_idx, n):
        # Get aligned values
        pivot = pivot_w_aligned[i]
        r4 = r4_w_aligned[i]
        s4 = s4_w_aligned[i]
        current_atr = atr[i]
        current_close = close[i]
        current_volume = volume[i]
        current_vol_ma = vol_ma[i]
        
        # Skip if any value is NaN
        if (np.isnan(pivot) or np.isnan(r4) or np.isnan(s4) or 
            np.isnan(current_atr) or np.isnan(current_vol_ma)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.5x 20-period average
        vol_condition = current_volume > 1.5 * current_vol_ma
        
        if position == 0:
            # Long: Break above weekly R4 with volume
            if current_close > r4 and vol_condition:
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            
            # Short: Break below weekly S4 with volume
            elif current_close < s4 and vol_condition:
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit: Price falls back below weekly pivot OR ATR stop
            if current_close < pivot or current_close < entry_price - 2.5 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises back above weekly pivot OR ATR stop
            if current_close > pivot or current_close > entry_price + 2.5 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
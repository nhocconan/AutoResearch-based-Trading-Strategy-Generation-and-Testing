#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_3day_RangeBreakout_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # === 1d: 3-day range (highest high, lowest low) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    window = 3
    highest_high = pd.Series(high_1d).rolling(window=window, min_periods=window).max().values
    lowest_low = pd.Series(low_1d).rolling(window=window, min_periods=window).min().values
    
    # Align to 6h
    highest_high_aligned = align_htf_to_ltf(prices, df_1d, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_1d, lowest_low)
    
    # === 6h: Indicators ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stop loss
    high = prices['high'].values
    low = prices['low'].values
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(30, 20)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Get values
        hh = highest_high_aligned[i]
        ll = lowest_low_aligned[i]
        vol_ma_val = vol_ma[i]
        vol = volume[i]
        c = close[i]
        atr_val = atr[i]
        
        # Skip if any value is NaN
        if (np.isnan(hh) or np.isnan(ll) or np.isnan(vol_ma_val) or np.isnan(atr_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.5x 20-period average
        vol_condition = vol > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: break above 3-day high + volume
            if c > hh and vol_condition:
                signals[i] = 0.25
                position = 1
                entry_price = c
            # Short: break below 3-day low + volume
            elif c < ll and vol_condition:
                signals[i] = -0.25
                position = -1
                entry_price = c
        
        elif position == 1:
            # Long exit: close below 3-day low OR stop loss
            if c < ll or c < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above 3-day high OR stop loss
            if c > hh or c > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
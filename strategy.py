#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_1d_VolumeSpike_V1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d ATR ratio (current / 20-period average) for volatility expansion
    atr_ma_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_1d / atr_ma_20
    
    # Align ATR ratio to 4h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 2.0 x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        if np.isnan(atr_ratio_aligned[i]) or np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr_ratio_val = atr_ratio_aligned[i]
        
        # Volatility filter: ATR ratio > 1.5 (volatility expansion)
        vol_expansion = atr_ratio_val > 1.5
        
        # Volume filter
        volume_ok = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long: price breaks above upper Donchian band with volume and volatility expansion
            if price > high_roll[i] and volume_ok and vol_expansion:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band with volume and volatility expansion
            elif price < low_roll[i] and volume_ok and vol_expansion:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns to middle of Donchian channel or volatility contracts
            mid = (high_roll[i] + low_roll[i]) / 2
            if price < mid or atr_ratio_val < 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to middle of Donchian channel or volatility contracts
            mid = (high_roll[i] + low_roll[i]) / 2
            if price > mid or atr_ratio_val < 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
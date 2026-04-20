#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_TurtleTrader_15_48"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 48:
        return np.zeros(n)
    
    # === 1d: Donchian channels (15 and 48) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian 15 (short-term)
    dh_15 = pd.Series(high_1d).rolling(window=15, min_periods=15).max().values
    dl_15 = pd.Series(low_1d).rolling(window=15, min_periods=15).min().values
    
    # Donchian 48 (long-term)
    dh_48 = pd.Series(high_1d).rolling(window=48, min_periods=48).max().values
    dl_48 = pd.Series(low_1d).rolling(window=48, min_periods=48).min().values
    
    # Align to 6h timeframe
    dh_15_6h = align_htf_to_ltf(prices, df_1d, dh_15)
    dl_15_6h = align_htf_to_ltf(prices, df_1d, dl_15)
    dh_48_6h = align_htf_to_ltf(prices, df_1d, dh_48)
    dl_48_6h = align_htf_to_ltf(prices, df_1d, dl_48)
    
    # === 6h: Price and volume ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(dh_15_6h[i]) or np.isnan(dl_15_6h[i]) or
            np.isnan(dh_48_6h[i]) or np.isnan(dl_48_6h[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        dh_15_val = dh_15_6h[i]
        dl_15_val = dl_15_6h[i]
        dh_48_val = dh_48_6h[i]
        dl_48_val = dl_48_6h[i]
        vol_ma20_val = vol_ma20[i]
        
        if position == 0:
            # Long: Break above 15-day Donchian high with volume
            # Short: Break below 15-day Donchian low with volume
            if (high_val > dh_15_val and volume[i] > vol_ma20_val):
                signals[i] = 0.25
                position = 1
            elif (low_val < dl_15_val and volume[i] > vol_ma20_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Break below 48-day Donchian low
            if low_val < dl_48_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Break above 48-day Donchian high
            if high_val > dh_48_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
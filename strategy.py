#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Donchian20_Breakout_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d: ATR for volatility filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])  # First TR is inf (no previous close)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 4h: Donchian channel (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian upper/lower bands
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume ratio (current vs 20-period average)
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Get values
        close_val = close[i]
        highest_val = highest_20[i]
        lowest_val = lowest_20[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr_1d[i]  # 1d ATR (already aligned via index)
        
        # Skip if any value is NaN
        if (np.isnan(highest_val) or np.isnan(lowest_val) or 
            np.isnan(vol_ratio_val) or np.isnan(atr_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above 20-period high with volume confirmation
            if close_val > highest_val and vol_ratio_val > 1.8:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: Break below 20-period low with volume confirmation
            elif close_val < lowest_val and vol_ratio_val > 1.8:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        
        elif position == 1:
            # Long exit: Stop loss or mean reversion
            stop_price = entry_price - 2.5 * atr_val
            if close_val < stop_price or close_val < highest_val * 0.995:  # Failed breakout
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Stop loss or mean reversion
            stop_price = entry_price + 2.5 * atr_val
            if close_val > stop_price or close_val > lowest_val * 1.005:  # Failed breakdown
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
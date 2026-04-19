#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Choppiness_Breakout_With_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Choppiness Index (14-day)
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    hh = np.maximum.accumulate(high_1d)
    ll = np.minimum.accumulate(low_1d)
    hh_ll = np.concatenate([[np.nan], hh[:-1] - ll[:-1]])
    
    chop = 100 * np.log10(atr14 / hh_ll) / np.log10(14)
    chop[0:14] = np.nan
    
    # Align Choppiness to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Donchian channel (20-period) on 4h
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 30)
    
    for i in range(start_idx, n):
        if np.isnan(chop_aligned[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(vol_ma_30[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_30[i]
        
        # Volume filter
        volume_ok = vol > 1.8 * vol_ma
        
        # Choppiness filter: trending when CHOP < 38.2
        trending = chop_aligned[i] < 38.2
        
        if position == 0:
            # Long: price breaks above Donchian high in trending market with volume
            if price > high_max[i] and trending and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low in trending market with volume
            elif price < low_min[i] and trending and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price falls below Donchian low or market becomes choppy
            if price < low_min[i] or chop_aligned[i] >= 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price rises above Donchian high or market becomes choppy
            if price > high_max[i] or chop_aligned[i] >= 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
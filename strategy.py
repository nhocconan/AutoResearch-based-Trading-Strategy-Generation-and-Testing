#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeFilter
Hypothesis: In both bull and bear markets, price tends to continue in the direction of the weekly pivot trend after breaking Donchian(20) channels. Weekly pivot provides higher timeframe bias (bullish if above weekly pivot, bearish if below). Volume surge confirms institutional participation. 6h timeframe balances responsiveness with low frequency. Target 15-30 trades/year to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly pivot (directional bias)
    df_1w = get_htf_data(prices, '1w')
    whe = df_1w['high'].values
    wol = df_1w['low'].values
    wcl = df_1w['close'].values
    w_pivot = (whe + wol + wcl) / 3.0
    w_pivot_aligned = align_htf_to_ltf(prices, df_1w, w_pivot)
    
    # Donchian channels (20-period)
    lookback = 20
    highest = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0 flat, 1 long, -1 short
    
    start_idx = max(lookback, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(w_pivot_aligned[i]) or 
            np.isnan(highest[i]) or 
            np.isnan(lowest[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        pivot = w_pivot_aligned[i]
        upper = highest[i]
        lower = lowest[i]
        vol_ok = volume_confirm[i]
        
        if position == 0:
            # Long: break above upper channel, above weekly pivot, volume confirmation
            if price > upper and price > pivot and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: break below lower channel, below weekly pivot, volume confirmation
            elif price < lower and price < pivot and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: break below lower channel or below weekly pivot
            if price < lower or price < pivot:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: break above upper channel or above weekly pivot
            if price > upper or price > pivot:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeFilter"
timeframe = "6h"
leverage = 1.0
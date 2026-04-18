#!/usr/bin/env python3
"""
4h_Donchian_Breakout_20_1dVolRatio_ATRStop_v1
Hypothesis: Donchian(20) breakouts on 4h with 1d volume ratio filter and ATR stoploss capture breakout momentum in both bull and bear markets.
Volume ratio filters for conviction, ATR stoploss manages risk. Designed for low trade frequency (15-25/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) on 4h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d volume ratio: current 4h volume vs 1d average volume per 4h bar
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    # Approximate: 1d volume / 6 (since 6*4h = 1d)
    vol_1d_per_4h = vol_1d / 6.0
    vol_1d_per_4h_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_per_4h)
    vol_ratio = volume / vol_1d_per_4h_aligned  # >1.5 means above average 4h volume for the day
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 40
    
    for i in range(start_idx, n):
        if (np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or
            np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = high_20[i]
        lower = low_20[i]
        vol_r = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: break above upper Donchian with volume confirmation
            if price > upper and vol_r > 1.5:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below lower Donchian with volume confirmation
            elif price < lower and vol_r > 1.5:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: stoploss or mean reversion to midpoint
            if price <= entry_price - 2.0 * atr_val or price <= (upper + lower) / 2:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: stoploss or mean reversion to midpoint
            if price >= entry_price + 2.0 * atr_val or price >= (upper + lower) / 2:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian_Breakout_20_1dVolRatio_ATRStop_v1"
timeframe = "4h"
leverage = 1.0
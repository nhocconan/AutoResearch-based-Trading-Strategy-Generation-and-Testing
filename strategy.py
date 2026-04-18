#!/usr/bin/env python3
"""
4h Donchian Breakout + Volume Spike + 1d ATR Filter (Long Only)
Hypothesis: In trending markets, price breaks above/below Donchian channels with volume confirmation capture momentum.
ATR filter ensures we only trade when volatility is sufficient (avoid chop). Long-only to avoid 2022 short whipsaw.
Designed for low trade frequency (<50/year) with strong edge in both bull (breakouts) and bear (avoid shorts) markets.
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
    
    # Donchian channel (20-period) on 4h
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = high_roll
    donchian_low = low_roll
    
    # 1d ATR for volatility filter (avoid chop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume spike (2x 4-period average)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 60  # need enough history for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        atr_val = atr_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume spike and sufficient volatility
            if (price > upper and 
                volume_spike[i] and 
                atr_val > 0.01 * price):  # ATR > 1% of price to avoid chop
                signals[i] = 0.25
                position = 1
                entry_price = price
        
        elif position == 1:
            # Maintain long position
            signals[i] = 0.25
            # Exit: price breaks below lower Donchian OR ATR drops too low (chop)
            if price < lower or atr_val < 0.005 * price:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian_Breakout_Volume_Spike_1dATRFilter"
timeframe = "4h"
leverage = 1.0
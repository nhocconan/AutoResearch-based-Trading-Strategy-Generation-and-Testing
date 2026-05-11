#!/usr/bin/env python3
"""
1h_Range_Breakout_Filtered_v1
Hypothesis: In ranging markets, price breaks out of consolidation with volume;
in trending markets, pullbacks to value areas offer entries. Uses 4h Donchian
channels for trend direction and 1d VWAP for value area, with 1h for precise
entry timing. Volume filter ensures breakouts have conviction.
Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe.
"""

name = "1h_Range_Breakout_Filtered_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4H Data for Trend Direction (Donchian Channel) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian Channel (20-period)
    upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align to 1h
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    
    # === 1D Data for Value Area (VWAP) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate VWAP for each day
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d = vwap_1d.values
    
    # Align VWAP to 1h
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # === 1H Volume Filter ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Start after warmup
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        # Session filter
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any required data is invalid
        if (np.isnan(upper_4h_aligned[i]) or 
            np.isnan(lower_4h_aligned[i]) or 
            np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above 4h upper Donchian with volume, above 1d VWAP (uptrend continuation or range breakout)
            if (close[i] > upper_4h_aligned[i] and 
                volume[i] > vol_ma[i] * 1.5 and 
                close[i] > vwap_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: break below 4h lower Donchian with volume, below 1d VWAP (downtrend continuation or range breakdown)
            elif (close[i] < lower_4h_aligned[i] and 
                  volume[i] > vol_ma[i] * 1.5 and 
                  close[i] < vwap_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price falls below 4h lower Donchian or below 1d VWAP with volume
            if close[i] < lower_4h_aligned[i] or (close[i] < vwap_1d_aligned[i] and volume[i] > vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20  # maintain position
        elif position == -1:
            # Short exit: price rises above 4h upper Donchian or above 1d VWAP with volume
            if close[i] > upper_4h_aligned[i] or (close[i] > vwap_1d_aligned[i] and volume[i] > vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20  # maintain position
    
    return signals
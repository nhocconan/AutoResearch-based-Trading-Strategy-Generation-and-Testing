#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_Range_Reversion_with_Volume_Filter
Hypothesis: In range-bound markets (common in 2025-2026), price reverts to the mean from Camarilla H3/L3 levels. 
Buy near L3 with volume confirmation, sell near H3. Uses 1d pivot levels and 6h trend filter to avoid counter-trend trades.
Designed for ~25 trades/year to minimize fee drag and work in both bull and bear markets via mean reversion in ranges.
"""

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
    
    # Calculate Camarilla levels from previous day (1d timeframe)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Camarilla calculation: H3/L3 from previous day
    # H3 = close + 1.1 * (high - low) / 4
    # L3 = close - 1.1 * (high - low) / 4
    camarilla_range = high_1d - low_1d
    h3_1d = close_1d + (1.1 * camarilla_range) / 4
    l3_1d = close_1d - (1.1 * camarilla_range) / 4
    
    # Align to 4h timeframe (use previous day's levels)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # 6h EMA trend filter (avoid strong counter-trend trades)
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    ema_6h = pd.Series(close_6h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_6h_aligned = align_htf_to_ltf(prices, df_6h, ema_6h)
    
    # Volume confirmation: >1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(ema_6h_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        h3 = h3_aligned[i]
        l3 = l3_aligned[i]
        ema_6h_val = ema_6h_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price near L3 (within 0.2%) with volume and above 6h EMA (avoid strong downtrend)
            if price <= l3 * 1.002 and vol_spike and price > ema_6h_val:
                signals[i] = 0.25
                position = 1
            # Short: price near H3 (within 0.2%) with volume and below 6h EMA (avoid strong uptrend)
            elif price >= h3 * 0.998 and vol_spike and price < ema_6h_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price reaches midpoint or breaks above H3
            midpoint = (h3 + l3) / 2
            if price >= midpoint or price > h3 * 1.002:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price reaches midpoint or breaks below L3
            midpoint = (h3 + l3) / 2
            if price <= midpoint or price < l3 * 0.998:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_Pivot_Range_Reversion_with_Volume_Filter"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
# 4h_daily_camarilla_pivot_volume_spike_v2
# Hypothesis: 4h strategies based on daily Camarilla pivot levels with volume spike confirmation work in both bull and bear markets.
# Long: price touches or breaks above Camarilla H3 level with volume > 2.0x 20-period average
# Short: price touches or breaks below Camarilla L3 level with volume > 2.0x 20-period average
# Exit: price reverts to Camarilla Pivot (midpoint) level or ATR-based stoploss (1.5x ATR)
# Uses 4h primary timeframe with 1d HTF for Camarilla pivot calculation.
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_daily_camarilla_pivot_volume_spike_v2"
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
    
    # Calculate ATR(14) for stoploss
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.mean(tr[i-14:i])
    
    # Calculate volume ratio (current vs 20-period average)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    vol_ratio = np.where(vol_sma > 0, volume / vol_sma, 0)
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    camarilla_p = np.full(len(df_1d), np.nan)
    camarilla_h3 = np.full(len(df_1d), np.nan)
    camarilla_l3 = np.full(len(df_1d), np.nan)
    camarilla_h4 = np.full(len(df_1d), np.nan)
    camarilla_l4 = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i == 0 or np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i]):
            continue
        diff = high_1d[i] - low_1d[i]
        camarilla_p[i] = (high_1d[i] + low_1d[i] + close_1d[i]) / 3.0
        camarilla_h3[i] = camarilla_p[i] + diff * 1.1 / 4.0
        camarilla_l3[i] = camarilla_p[i] - diff * 1.1 / 4.0
        camarilla_h4[i] = camarilla_p[i] + diff * 1.1 / 2.0
        camarilla_l4[i] = camarilla_p[i] - diff * 1.1 / 2.0
    
    # Align 1d Camarilla levels to 4h timeframe
    camarilla_p_aligned = align_htf_to_ltf(prices, df_1d, camarilla_p)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(50, n):
        vol_r = vol_ratio[i]
        price = close[i]
        
        if np.isnan(vol_r) or np.isnan(camarilla_p_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reverts to pivot OR stoploss hit (1.5x ATR below entry)
            if price <= camarilla_p_aligned[i] or price <= entry_price - 1.5 * atr_stop:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reverts to pivot OR stoploss hit (1.5x ATR above entry)
            if price >= camarilla_p_aligned[i] or price >= entry_price + 1.5 * atr_stop:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: price touches/breaks above H3 with volume spike
            if price >= camarilla_h3_aligned[i] and vol_r > 2.0:
                position = 1
                entry_price = price
                atr_stop = atr[i]
                signals[i] = 0.25
            # Short entry: price touches/breaks below L3 with volume spike
            elif price <= camarilla_l3_aligned[i] and vol_r > 2.0:
                position = -1
                entry_price = price
                atr_stop = atr[i]
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
# 12h_daily_camarilla_pivot_volume_regime_v1
# Hypothesis: 12h strategies using daily Camarilla pivot levels with volume confirmation and chop regime filter work in both bull and bear markets.
# Long: price touches/breaks above Camarilla H3 level with volume > 2.0x 20-period average and CHOP > 61.8 (range)
# Short: price touches/breaks below Camarilla L3 level with volume > 2.0x 20-period average and CHOP > 61.8 (range)
# Exit: price reverts to Camarilla Pivot (midpoint) level
# Uses 12h primary timeframe with 1d HTF for Camarilla pivot and CHOP calculation.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_daily_camarilla_pivot_volume_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss with min_periods
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.mean(tr[i-14:i])
    
    # Calculate volume ratio (current vs 20-period average) with min_periods
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    vol_ratio = np.where(vol_sma > 0, volume / vol_sma, 0)
    
    # Get 1d data for Camarilla pivot levels and CHOP
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
    
    for i in range(len(df_1d)):
        if i == 0 or np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i]):
            continue
        diff = high_1d[i] - low_1d[i]
        camarilla_p[i] = (high_1d[i] + low_1d[i] + close_1d[i]) / 3.0
        camarilla_h3[i] = camarilla_p[i] + diff * 1.1 / 4.0
        camarilla_l3[i] = camarilla_p[i] - diff * 1.1 / 4.0
    
    # Calculate CHOP(14) for 1d timeframe
    chop = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        true_range = np.zeros(14)
        for j in range(14):
            idx = i - 13 + j
            tr_val = max(high_1d[idx] - low_1d[idx], 
                         abs(high_1d[idx] - close_1d[idx-1]), 
                         abs(low_1d[idx] - close_1d[idx-1]))
            true_range[j] = tr_val
        atr_14 = np.mean(true_range)
        max_high = np.max(high_1d[i-13:i+1])
        min_low = np.min(low_1d[i-13:i+1])
        if max_high - min_low > 0:
            chop[i] = 100 * np.log10(atr_14 * 14 / (max_high - min_low)) / np.log10(10)
        else:
            chop[i] = 50.0  # neutral when no range
    
    # Align 1d Camarilla levels and CHOP to 12h timeframe
    camarilla_p_aligned = align_htf_to_ltf(prices, df_1d, camarilla_p)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(50, n):
        vol_r = vol_ratio[i]
        price = close[i]
        ch = chop_aligned[i]
        
        # Skip if any required data is NaN
        if np.isnan(vol_r) or np.isnan(camarilla_p_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or np.isnan(ch):
            # Hold current position if any, otherwise flat
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            continue
        
        if position == 1:  # Long position
            # Exit: price reverts to pivot
            if price <= camarilla_p_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reverts to pivot
            if price >= camarilla_p_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: price touches/breaks above H3 with volume spike and chop > 61.8 (range)
            if price >= camarilla_h3_aligned[i] and vol_r > 2.0 and ch > 61.8:
                position = 1
                entry_price = price
                signals[i] = 0.25
            # Short entry: price touches/breaks below L3 with volume spike and chop > 61.8 (range)
            elif price <= camarilla_l3_aligned[i] and vol_r > 2.0 and ch > 61.8:
                position = -1
                entry_price = price
                signals[i] = -0.25
    
    return signals
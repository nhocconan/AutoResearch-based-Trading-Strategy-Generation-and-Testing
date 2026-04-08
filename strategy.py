#!/usr/bin/env python3
# 12h_weekly_camarilla_pivot_volume_regime_v1
# Hypothesis: 12h strategies using weekly Camarilla pivot levels with volume confirmation and choppiness regime filter work in both bull and bear markets.
# Long: price touches/breaks above weekly Camarilla H3 level with volume > 2.0x 50-period average and CHOP > 61.8 (ranging market)
# Short: price touches/breaks below weekly Camarilla L3 level with volume > 2.0x 50-period average and CHOP > 61.8 (ranging market)
# Exit: price reverts to weekly Camarilla Pivot level or ATR-based stoploss (2.0x ATR)
# Uses 12h primary timeframe with 1w HTF for Camarilla pivot calculation and 1d HTF for choppiness filter.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_weekly_camarilla_pivot_volume_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(20) for stoploss with min_periods
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = np.full(n, np.nan)
    for i in range(20, n):
        atr[i] = np.mean(tr[i-20:i])
    
    # Calculate volume ratio (current vs 50-period average) with min_periods
    vol_sma = np.full(n, np.nan)
    for i in range(50, n):
        vol_sma[i] = np.mean(volume[i-50:i])
    vol_ratio = np.where(vol_sma > 0, volume / vol_sma, 0)
    
    # Get 1w data for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla pivot levels for each 1w bar
    camarilla_p_1w = np.full(len(df_1w), np.nan)
    camarilla_h3_1w = np.full(len(df_1w), np.nan)
    camarilla_l3_1w = np.full(len(df_1w), np.nan)
    
    for i in range(len(df_1w)):
        if i == 0 or np.isnan(high_1w[i]) or np.isnan(low_1w[i]) or np.isnan(close_1w[i]):
            continue
        diff = high_1w[i] - low_1w[i]
        camarilla_p_1w[i] = (high_1w[i] + low_1w[i] + close_1w[i]) / 3.0
        camarilla_h3_1w[i] = camarilla_p_1w[i] + diff * 1.1 / 4.0
        camarilla_l3_1w[i] = camarilla_p_1w[i] - diff * 1.1 / 4.0
    
    # Align 1w Camarilla levels to 12h timeframe
    camarilla_p_aligned = align_htf_to_ltf(prices, df_1w, camarilla_p_1w)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3_1w)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3_1w)
    
    # Get 1d data for choppiness regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for 1d
    tr_1d = np.zeros(len(df_1d))
    for i in range(1, len(df_1d)):
        tr_1d[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
    
    # Calculate ATR(14) for 1d
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_1d[i] = np.mean(tr_1d[i-14:i])
    
    # Calculate sum of true ranges over 14 periods for choppy market calculation
    sum_tr_14 = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        sum_tr_14[i] = np.sum(tr_1d[i-14:i])
    
    # Calculate max(high) - min(low) over 14 periods
    max_high_14 = np.full(len(df_1d), np.nan)
    min_low_14 = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        max_high_14[i] = np.max(high_1d[i-14:i])
        min_low_14[i] = np.min(low_1d[i-14:i])
    
    # Calculate Choppiness Index (CHOP)
    chop = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        if sum_tr_14[i] > 0 and max_high_14[i] > min_low_14[i]:
            chop[i] = 100 * np.log10(sum_tr_14[i] / (max_high_14[i] - min_low_14[i])) / np.log10(14)
        else:
            chop[i] = 50.0  # neutral value when calculation not possible
    
    # Align 1d Choppiness Index to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        vol_r = vol_ratio[i]
        price = close[i]
        ch = chop_aligned[i]
        
        # Skip if any required data is NaN
        if np.isnan(vol_r) or np.isnan(camarilla_p_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or np.isnan(atr[i]) or np.isnan(ch):
            # Hold current position if any, otherwise flat
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            continue
        
        if position == 1:  # Long position
            bars_since_entry += 1
            # Exit: price reverts to pivot OR stoploss hit (2.0x ATR below entry) OR min holding period (4 bars) passed
            if (price <= camarilla_p_aligned[i] or 
                price <= entry_price - 2.0 * atr_stop or 
                bars_since_entry >= 4):
                position = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            bars_since_entry += 1
            # Exit: price reverts to pivot OR stoploss hit (2.0x ATR above entry) OR min holding period (4 bars) passed
            if (price >= camarilla_p_aligned[i] or 
                price >= entry_price + 2.0 * atr_stop or 
                bars_since_entry >= 4):
                position = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            bars_since_entry = 0
            # Long entry: price touches/breaks above H3 with volume spike and choppy market (CHOP > 61.8)
            if price >= camarilla_h3_aligned[i] and vol_r > 2.0 and ch > 61.8:
                position = 1
                entry_price = price
                atr_stop = atr[i]
                signals[i] = 0.25
            # Short entry: price touches/breaks below L3 with volume spike and choppy market (CHOP > 61.8)
            elif price <= camarilla_l3_aligned[i] and vol_r > 2.0 and ch > 61.8:
                position = -1
                entry_price = price
                atr_stop = atr[i]
                signals[i] = -0.25
    
    return signals
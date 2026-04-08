#!/usr/bin/env python3
# 12h_daily_camarilla_pivot_volume_regime_v1
# Hypothesis: 12h strategies based on daily Camarilla pivot levels with volume confirmation and chop regime filter work in both bull and bear markets.
# Long: price touches/breaks above Camarilla H3 with volume > 1.8x 20-period average and chop < 61.8 (trending regime)
# Short: price touches/breaks below Camarilla L3 with volume > 1.8x 20-period average and chop < 61.8 (trending regime)
# Exit: price reverts to Camarilla Pivot level or ATR stoploss (2.0x ATR)
# Uses 12h primary timeframe with 1d HTF for Camarilla pivot and chop calculation.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_daily_camarilla_pivot_volume_regime_v1"
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
    
    # Get 1d data for Camarilla pivot levels and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
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
    
    # Calculate Choppiness Index (14) for regime filter
    chop = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        # True range sum over 14 periods
        tr_sum = 0.0
        for j in range(i-13, i+1):
            if j > 0:
                tr_j = max(high_1d[j] - low_1d[j], abs(high_1d[j] - close_1d[j-1]), abs(low_1d[j] - close_1d[j-1]))
            else:
                tr_j = high_1d[j] - low_1d[j]
            tr_sum += tr_j
        # Chop = 100 * log10(tr_sum / (max_high - min_low)) / log10(14)
        max_high = np.max(high_1d[i-13:i+1])
        min_low = np.min(low_1d[i-13:i+1])
        if tr_sum > 0 and (max_high - min_low) > 0:
            chop[i] = 100 * np.log10(tr_sum / (max_high - min_low)) / np.log10(14)
        else:
            chop[i] = 50.0  # neutral
    
    # Align 1d data to 12h timeframe
    camarilla_p_aligned = align_htf_to_ltf(prices, df_1d, camarilla_p)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(100, n):
        vol_r = vol_ratio[i]
        price = close[i]
        chop_val = chop_aligned[i]
        
        # Skip if any required data is NaN or invalid chop
        if np.isnan(vol_r) or np.isnan(camarilla_p_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or np.isnan(chop_val) or np.isnan(atr[i]):
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            continue
        
        # Regime filter: only trade in trending markets (chop < 61.8)
        in_trending_regime = chop_val < 61.8
        
        if position == 1:  # Long position
            # Exit: price reverts to pivot OR stoploss hit (2.0x ATR below entry)
            if price <= camarilla_p_aligned[i] or price <= entry_price - 2.0 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reverts to pivot OR stoploss hit (2.0x ATR above entry)
            if price >= camarilla_p_aligned[i] or price >= entry_price + 2.0 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: price touches/breaks above H3 with volume spike in trending regime
            if price >= camarilla_h3_aligned[i] and vol_r > 1.8 and in_trending_regime:
                position = 1
                entry_price = price
                atr_stop = atr[i]
                signals[i] = 0.25
            # Short entry: price touches/breaks below L3 with volume spike in trending regime
            elif price <= camarilla_l3_aligned[i] and vol_r > 1.8 and in_trending_regime:
                position = -1
                entry_price = price
                atr_stop = atr[i]
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
# 6h_weekly_pivot_breakout_volume_v1
# Hypothesis: 6h strategies based on weekly Camarilla pivot levels with volume confirmation work in both bull and bear markets.
# Weekly pivots provide stronger support/resistance than daily, reducing false breakouts.
# Long: price breaks above weekly H4 level with volume > 2.0x 20-period average
# Short: price breaks below weekly L4 level with volume > 2.0x 20-period average
# Exit: price reverts to weekly pivot level or ATR-based stop (2.0x ATR)
# Uses 6h primary timeframe with 1w HTF for weekly Camarilla pivot calculation.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Get 1w data for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla pivot levels for each 1w bar
    camarilla_p = np.full(len(df_1w), np.nan)
    camarilla_h3 = np.full(len(df_1w), np.nan)
    camarilla_l3 = np.full(len(df_1w), np.nan)
    camarilla_h4 = np.full(len(df_1w), np.nan)
    camarilla_l4 = np.full(len(df_1w), np.nan)
    
    for i in range(len(df_1w)):
        if i == 0 or np.isnan(high_1w[i]) or np.isnan(low_1w[i]) or np.isnan(close_1w[i]):
            continue
        diff = high_1w[i] - low_1w[i]
        camarilla_p[i] = (high_1w[i] + low_1w[i] + close_1w[i]) / 3.0
        camarilla_h3[i] = camarilla_p[i] + diff * 1.1 / 4.0
        camarilla_l3[i] = camarilla_p[i] - diff * 1.1 / 4.0
        camarilla_h4[i] = camarilla_p[i] + diff * 1.1 / 2.0
        camarilla_l4[i] = camarilla_p[i] - diff * 1.1 / 2.0
    
    # Align 1w Camarilla levels to 6h timeframe
    camarilla_p_aligned = align_htf_to_ltf(prices, df_1w, camarilla_p)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(100, n):
        vol_r = vol_ratio[i]
        price = close[i]
        
        if np.isnan(vol_r) or np.isnan(camarilla_p_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reverts to pivot OR stoploss hit (2.0x ATR below entry)
            if price <= camarilla_p_aligned[i] or price <= entry_price - 2.0 * atr_stop:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reverts to pivot OR stoploss hit (2.0x ATR above entry)
            if price >= camarilla_p_aligned[i] or price >= entry_price + 2.0 * atr_stop:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: price breaks above H4 with volume spike
            if price > camarilla_h4_aligned[i] and vol_r > 2.0:
                position = 1
                entry_price = price
                atr_stop = atr[i]
                signals[i] = 0.25
            # Short entry: price breaks below L4 with volume spike
            elif price < camarilla_l4_aligned[i] and vol_r > 2.0:
                position = -1
                entry_price = price
                atr_stop = atr[i]
                signals[i] = -0.25
    
    return signals
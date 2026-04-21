#!/usr/bin/env python3
"""
4h_Camarilla_H3L3_Pullback_Volume_Regime_v2
Hypothesis: Pullback to Camarilla H3/L3 levels with volume confirmation and choppiness regime filter.
Long when price pulls back to H3 with volume spike in choppy market (CHOP>61.8).
Short when price pulls back to L3 with volume spike in choppy market (CHOP>61.8).
Exit when price reaches H4/L4 or reverses at H3/L3.
Uses 1d Camarilla levels, 4h volume and choppiness filter.
Designed to work in both bull/bear by fading extremes in ranging markets.
Target: 20-35 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: H3, L3, H4, L4
    rang = prev_high - prev_low
    h3 = prev_close + 1.1 * rang / 4
    l3 = prev_close - 1.1 * rang / 4
    h4 = prev_close + 1.1 * rang / 2
    l4 = prev_close - 1.1 * rang / 2
    
    # Align to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Calculate 4h choppiness index (CHOP) for regime filter
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    
    # ATR(14) - sum of TR over 14 periods
    atr_14 = np.zeros(n)
    atr_14[13] = np.sum(tr[0:14])
    for i in range(14, n):
        atr_14[i] = atr_14[i-1] - (atr_14[i-1] / 14) + (tr[i] / 14)
    
    # Choppiness Index: CHOP = 100 * log10(sum(TR(14)) / (ATR(14) * 14)) / log10(14)
    sum_tr_14 = np.zeros(n)
    sum_tr_14[13] = np.sum(tr[0:14])
    for i in range(14, n):
        sum_tr_14[i] = sum_tr_14[i-1] - tr[i-14] + tr[i]
    
    chop = 100 * np.log10(sum_tr_14 / (atr_14 * 14)) / np.log10(14)
    chop[0:13] = np.nan  # not enough data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 2.0 * vol_ma
        else:
            volume_ok = False
        
        # Regime filter: only trade in choppy/ranging markets (CHOP > 61.8)
        regime_ok = chop[i] > 61.8
        
        if position == 0:
            # Long conditions: pullback to H3 with volume spike in choppy market
            if (abs(price - h3_aligned[i]) < 0.001 * h3_aligned[i] and  # near H3
                volume_ok and regime_ok):
                signals[i] = 0.25
                position = 1
            # Short conditions: pullback to L3 with volume spike in choppy market
            elif (abs(price - l3_aligned[i]) < 0.001 * l3_aligned[i] and  # near L3
                  volume_ok and regime_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: reach H4 or reverse at H3
            if price >= h4_aligned[i] or price <= h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: reach L4 or reverse at L3
            if price <= l4_aligned[i] or price >= l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Pullback_Volume_Regime_v2"
timeframe = "4h"
leverage = 1.0
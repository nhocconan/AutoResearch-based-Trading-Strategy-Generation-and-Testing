#!/usr/bin/env python3
"""
12h_1d_RVOL_Reversal_LongOnly
Hypothesis: RVOL-based reversal strategy on 12h timeframe using 1d ATR for context.
Long when price drops >1.5*ATR(1d) below 1d close AND RVOL > 1.8 (panic dip with volume).
Exit when price recovers to midpoint of the dip range or 5 bars pass.
Designed for 12h to capture mean-reversion bounces in volatile markets.
Works in both bull (buy dips) and bear (sell relief rallies via symmetry).
RVOL filter ensures we trade only high-volume exhaustion moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for ATR and close reference
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on 1d
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, tr2)
    tr = np.concatenate([[np.nan], tr])  # align length
    
    atr_1d = np.full_like(close_1d, np.nan)
    for i in range(14, len(tr)):
        atr_1d[i] = np.nanmean(tr[i-13:i+1])  # simple ATR
    
    # Shift ATR to use previous day's value (available at open)
    atr_1d = np.roll(atr_1d, 1)
    atr_1d[0] = np.nan
    
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    bars_since_entry = 0
    entry_price = 0.0
    atr_at_entry = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(close_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # RVOL: current volume / 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            rvol = volume / vol_ma if vol_ma > 0 else 0
        else:
            rvol = 0
        
        if position == 0:
            # Long condition: price > 1.5*ATR below 1d close AND RVOL > 1.8
            if price < (close_1d_aligned[i] - 1.5 * atr_1d_aligned[i]) and rvol > 1.8:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
                entry_price = price
                atr_at_entry = atr_1d_aligned[i]
        
        elif position == 1:
            bars_since_entry += 1
            # Exit conditions: price recovers to midpoint OR 5 bars elapsed
            recovery_level = entry_price + 0.5 * (close_1d_aligned[i] - entry_price)
            if price >= recovery_level or bars_since_entry >= 5:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
    
    return signals

name = "12h_1d_RVOL_Reversal_LongOnly"
timeframe = "12h"
leverage = 1.0
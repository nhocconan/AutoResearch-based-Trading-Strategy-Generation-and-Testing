#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_Volume_Regime_V1
Hypothesis: 12h Camarilla R1/S1 breakout with volume confirmation (>2.0x 20-bar volume MA) and 1d chop regime filter (CHOP > 61.8 for mean reversion).
Long when price breaks above R1 with volume + chop regime; short when breaks below S1 with volume + chop regime.
ATR-based trailing stop (2.5x ATR) to manage risk. Works in ranging markets via mean reversion at pivot levels.
Position size 0.25 balances risk/return. Target 12-37 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Camarilla pivot levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = pivot + (high_1d - low_1d) * 1.1 / 12.0
    s1 = pivot - (high_1d - low_1d) * 1.1 / 12.0
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 1d Choppiness Index regime filter ===
    # CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending
    high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    atr_14 = pd.Series(high_1d - low_1d).rolling(window=14, min_periods=14).sum().values
    denom = np.maximum(atr_14, 1e-10)
    chop = 100 * np.log10(denom / np.maximum(high_14 - low_14, 1e-10)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 12h Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ATR (14-period) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(chop_aligned[i]) 
            or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().iloc[i] if i >= 20 else np.nan
        
        if np.isnan(vol_ma):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        vol_ok = vol > 2.0 * vol_ma  # volume confirmation
        chop_ok = chop_aligned[i] > 61.8  # ranging regime for mean reversion
        
        if position == 0:
            # Long: price breaks above R1 + volume confirmation + chop regime (ranging)
            if price > r1_aligned[i] and vol_ok and chop_ok:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: price breaks below S1 + volume confirmation + chop regime (ranging)
            elif price < s1_aligned[i] and vol_ok and chop_ok:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        
        elif position == 1:
            # Update highest since entry
            highest_since_entry = max(highest_since_entry, price)
            # ATR trailing stop: exit if price drops 2.5*ATR from highest since entry
            if price < highest_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest since entry
            lowest_since_entry = min(lowest_since_entry, price)
            # ATR trailing stop: exit if price rises 2.5*ATR from lowest since entry
            if price > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_Volume_Regime_V1"
timeframe = "12h"
leverage = 1.0
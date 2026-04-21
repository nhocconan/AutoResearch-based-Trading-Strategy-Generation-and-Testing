#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_Regime_Volume_ATRFilter_V1
Hypothesis: Donchian(20) breakout on 4h with chop regime filter (CHOP>61.8 = range) and volume confirmation.
In ranging markets (CHOP>61.8), fade breakouts; in trending markets (CHOP<38.2), follow breakouts.
Uses ATR-based stoploss and discrete position sizing (0.25) to minimize fee churn.
Target: 25-50 trades/year per symbol (100-200 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ATR on daily
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 14-period rolling max/min for chop denominator
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr14 / (max_high - min_low)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian(20) on 4h prices
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # ATR(20) for stoploss on 4h
    tr_4h1 = np.abs(high_4h[1:] - low_4h[1:])
    tr_4h2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr_4h3 = np.abs(low_4h[1:] - close_4h[:-1])
    close_4h = prices['close'].values
    tr_4h = np.concatenate([[np.nan], np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))])
    atr_20 = pd.Series(tr_4h).rolling(window=20, min_periods=20).mean().values
    
    # Volume filter: current volume > 1.3 * 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > 1.3 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(atr_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol_ok = volume_ok[i] if i < len(volume_ok) else False
        chop_val = chop_aligned[i]
        
        # Determine regime: chop > 61.8 = range (mean revert), chop < 38.2 = trending (follow)
        is_range = chop_val > 61.8
        is_trending = chop_val < 38.2
        
        if position == 0:
            # Long conditions
            long_breakout = price > donch_high[i-1]  # break above upper band
            # In trending market: follow breakout; in ranging market: fade breakout (so short)
            if is_trending and long_breakout and vol_ok:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # In ranging market: short on breakdown of lower band
            elif is_range and price < donch_low[i-1] and vol_ok:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: stoploss or time-based
            stop_price = entry_price - 2.5 * atr_20[i]
            if price < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stoploss or time-based
            stop_price = entry_price + 2.5 * atr_20[i]
            if price > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_Regime_Volume_ATRFilter_V1"
timeframe = "4h"
leverage = 1.0
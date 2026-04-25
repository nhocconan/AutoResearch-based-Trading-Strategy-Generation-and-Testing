#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyATR_Breakout_v1
Hypothesis: Trade 6h Donchian(20) breakouts confirmed by weekly ATR expansion and volume spike.
Only long when price breaks above Donchian(20) high AND weekly ATR(14) > 1.2 * weekly ATR(50) AND volume > 2.0 * ATR6h.
Only short when price breaks below Donchian(20) low AND same weekly ATR expansion condition AND volume > 2.0 * ATR6h.
Weekly ATR expansion identifies periods of institutional participation, filtering low-volume false breakouts.
Works in both bull and bear markets by trading breakouts with momentum confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for ATR expansion filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:  # need enough for ATR(50)
        return np.zeros(n)
    
    # Calculate weekly ATR(14) and ATR(50) for expansion filter
    tr1_w = np.maximum(df_1w['high'][1:] - df_1w['low'][1:], np.abs(df_1w['high'][1:] - df_1w['close'][:-1]))
    tr2_w = np.maximum(np.abs(df_1w['low'][1:] - df_1w['close'][:-1]), tr1_w)
    tr_w = np.concatenate([[np.inf], tr2_w])
    atr14_w = pd.Series(tr_w).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr50_w = pd.Series(tr_w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly ATR values to 6h timeframe
    atr14_w_aligned = align_htf_to_ltf(prices, df_1w, atr14_w)
    atr50_w_aligned = align_htf_to_ltf(prices, df_1w, atr50_w)
    
    # Weekly ATR expansion: ATR(14) > 1.2 * ATR(50)
    weekly_atr_expansion = atr14_w_aligned > (1.2 * atr50_w_aligned)
    
    # Calculate Donchian(20) on 6h data
    lookback = 20
    highest = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate ATR for volume confirmation (using 6h data)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian(20), ATR(14), and weekly ATR
    start_idx = max(lookback, 14, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest[i]) or np.isnan(lowest[i]) or 
            np.isnan(atr[i]) or np.isnan(atr14_w_aligned[i]) or 
            np.isnan(atr50_w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0 * ATR (stronger filter)
        volume_confirm = volume[i] > (2.0 * atr[i])
        
        # Weekly ATR expansion filter
        atr_expand = weekly_atr_expansion[i]
        
        if position == 0:
            # Long setup: price breaks above Donchian high AND volume confirm AND ATR expansion
            long_setup = (close[i] > highest[i]) and volume_confirm and atr_expand
            
            # Short setup: price breaks below Donchian low AND volume confirm AND ATR expansion
            short_setup = (close[i] < lowest[i]) and volume_confirm and atr_expand
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below Donchian low OR ATR expansion ends
            if (close[i] < lowest[i]) or (not atr_expand):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above Donchian high OR ATR expansion ends
            if (close[i] > highest[i]) or (not atr_expand):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_WeeklyATR_Breakout_v1"
timeframe = "6h"
leverage = 1.0
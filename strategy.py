#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeChopFilter_v1
Hypothesis: Trade Donchian(20) breakouts on 4h with volume spike and chop regime filter.
Long when price breaks above upper Donchian + volume > 1.5x average + chop > 61.8 (range).
Short when price breaks below lower Donchian + volume > 1.5x average + chop > 61.8 (range).
Exit on opposite Donchian touch or chop < 38.2 (trend) to avoid whipsaw.
Position size: 0.25. Target: 20-50 trades/year to stay under 400 total 4h trades.
Works in bull (breakouts with volume) and bear (breakdowns with volume) markets.
Chop filter avoids false breakouts in strong trends where Donchian alone fails.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d chop regime (EWMA-based for efficiency)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR(14) using EWMA
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Chop = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(14)
    # Using rolling window of 14
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr_sum / (max_high - min_low + 1e-10)) / np.log10(14)
    chop = np.where((max_high - min_low) > 0, chop_raw, 50.0)  # default to 50 when range=0
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Donchian(20) on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume spike: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian(20) and volume MA
    start_idx = lookback
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        vol_spike = volume_spike[i]
        current_chop = chop_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above upper Donchian + volume spike + chop > 61.8 (range)
            long_setup = (close[i] > highest_high[i]) and vol_spike and (current_chop > 61.8)
            
            # Short setup: price breaks below lower Donchian + volume spike + chop > 61.8 (range)
            short_setup = (close[i] < lowest_low[i]) and vol_spike and (current_chop > 61.8)
            
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
            # Exit: price touches lower Donchian (stop) OR chop < 38.2 (trend) to avoid whipsaw
            if (close[i] <= lowest_low[i]) or (current_chop < 38.2):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches upper Donchian (stop) OR chop < 38.2 (trend) to avoid whipsaw
            if (close[i] >= highest_high[i]) or (current_chop < 38.2):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeChopFilter_v1"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) Breakout with Volume Spike and Choppiness Regime Filter
- Entry: Price breaks Donchian(20) high/low + volume > 2.0x 20-period MA + choppy market (CHOP > 61.8)
- Exit: Price closes back inside Donchian(20) channel
- Uses discrete position sizing (0.25) to minimize fee churn
- Choppiness filter ensures trades occur in ranging markets where breakouts are more reliable
- Target: 20-30 trades/year per symbol (<120 total over 4 years) to avoid fee drag
- Works in both bull and bear markets via regime filter and volume confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR for Choppiness indicator (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Choppiness Index: CHOP = 100 * log10(sum(ATR(14)) / (max(high) - min(low)) over period) / log10(N)
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    sum_atr = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    chop = 100 * (np.log10(sum_atr / (max_high - min_low)) / np.log10(14))
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian(20) channels from 1d data
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # need ATR(14)+14 for CHOP, Donchian(20), vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high + volume spike + choppy market (CHOP > 61.8)
            if (close[i] > donch_high_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i] and 
                chop_aligned[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low + volume spike + choppy market (CHOP > 61.8)
            elif (close[i] < donch_low_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i] and 
                  chop_aligned[i] > 61.8):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price closes back inside Donchian channel
            exit_signal = False
            if position == 1:
                # Exit long when close < Donchian low
                if close[i] < donch_low_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when close > Donchian high
                if close[i] > donch_high_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0
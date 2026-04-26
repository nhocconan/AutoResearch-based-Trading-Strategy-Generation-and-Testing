#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_ChopRegime_v1
Hypothesis: Donchian(20) breakouts on 4h with volume confirmation and choppiness regime filter capture strong momentum moves while avoiding whipsaws in ranging markets. Volume > 1.5x 20-period EMA confirms breakout strength. Choppiness Index > 61.8 indicates ranging (mean reversion), < 38.2 indicates trending (breakout continuation). Designed for 20-50 trades/year to minimize fee drag while maintaining edge in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF regime filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ATR(14) for Choppiness Index
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([np.array([0.0]), tr])  # align length
    
    # ATR(14)
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Choppiness Index: 100 * log10(sum(atr_14,14) / (max(high,14)-min(low,14))) / log10(14)
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    denominator = max_high - min_low
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    chop = 100 * np.log10(atr_sum / denominator) / np.log10(14)
    
    # Align Chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian(20) on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume spike detection on 4h (volume > 1.5x 20-period EMA)
    volume_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (volume_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(lookback, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(chop_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime filter: Chop < 38.2 = trending (favor breakouts), Chop > 61.8 = ranging (avoid breakouts)
        trending_regime = chop_aligned[i] < 38.2
        ranging_regime = chop_aligned[i] > 61.8
        
        # Long logic: price breaks above Donchian high + volume spike + trending regime
        if high[i] > highest_high[i] and volume_spike[i] and trending_regime:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: price breaks below Donchian low + volume spike + trending regime
        elif low[i] < lowest_low[i] and volume_spike[i] and trending_regime:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: price returns to opposite Donchian level or regime shifts to ranging
        elif position == 1 and (low[i] < lowest_low[i] or ranging_regime):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (high[i] > highest_high[i] or ranging_regime):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0
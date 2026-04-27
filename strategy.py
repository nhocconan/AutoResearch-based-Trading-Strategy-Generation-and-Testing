#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_Regime
Hypothesis: 4h Donchian(20) breakout with volume confirmation and chop regime filter.
Long when price breaks above upper Donchian channel + volume spike + chop > 61.8 (range) for mean reversion.
Short when price breaks below lower Donchian channel + volume spike + chop > 61.8 (range) for mean reversion.
Uses ATR-based trailing stop for risk control. Designed for 20-40 trades/year on 4h to minimize fee drag.
Works in both bull and bear markets by using chop regime to identify mean-reversion opportunities.
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
    
    # Calculate ATR (20-period) for Donchian channels and stops
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr0 = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr0], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Choppiness Index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low over 14))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = hh_14 - ll_14
    chop = 100 * np.log10(sum_atr_14 / np.log10(range_14)) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((range_14 > 0) & (sum_atr_14 > 0), chop, 50.0)
    chop = np.where(np.isnan(chop), 50.0, chop)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: need enough for ATR, Donchian, CHOP, volume average
    start_idx = max(100, 20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(chop[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        size = 0.25  # 25% position size
        
        # Regime filter: only trade in choppy markets (CHOP > 61.8) for mean reversion
        in_choppy_regime = chop[i] > 61.8
        
        if position == 0:
            # Flat - look for entry: Donchian breakout with volume spike in choppy regime
            # Long: price breaks above upper Donchian AND volume spike AND choppy regime
            # Short: price breaks below lower Donchian AND volume spike AND choppy regime
            long_breakout = close_val > highest_high[i]
            short_breakout = close_val < lowest_low[i]
            
            if long_breakout and vol_spike and in_choppy_regime:
                signals[i] = size
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_breakout and vol_spike and in_choppy_regime:
                signals[i] = -size
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
        elif position == 1:
            # Long - update highest and check exit conditions
            highest_since_entry = max(highest_since_entry, close_val)
            # Exit when: price breaks below lower Donchian (failed breakout) OR ATR trailing stop hit
            if close_val < lowest_low[i] or close_val < highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - update lowest and check exit conditions
            lowest_since_entry = min(lowest_since_entry, close_val)
            # Exit when: price breaks above upper Donchian (failed breakout) OR ATR trailing stop hit
            if close_val > highest_high[i] or close_val > lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0
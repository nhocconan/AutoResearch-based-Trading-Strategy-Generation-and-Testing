#!/usr/bin/env python3
"""
4h_Vortex_Trend_Plus_Volume_Regime
Strategy: 4h Vortex Indicator trend + volume spike + chop regime filter.
Long: VI+ > VI- + volume > 1.5x 20-period avg + CHOP > 61.8 (range)
Short: VI- > VI+ + volume > 1.5x 20-period avg + CHOP > 61.8 (range)
Exit: Trend reversal (VI cross) or volume/volatility fails
Position size: 0.25
Designed to capture trend moves in ranging markets with volume confirmation.
Timeframe: 4h
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
    
    # Calculate True Range components
    tr1 = high - low
    tr2 = np.abs(np.concatenate([[high[0]], high[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(np.concatenate([[low[0]], low[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate Vortex Indicator components
    vm_plus = np.abs(high - np.concatenate([[low[0]], low[:-1]]))
    vm_minus = np.abs(low - np.concatenate([[high[0]], high[:-1]]))
    
    # Sum over 14 periods
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vm_plus14 = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus14 = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    
    # VI+ and VI-
    vi_plus = np.divide(vm_plus14, tr14, out=np.zeros_like(vm_plus14), where=tr14!=0)
    vi_minus = np.divide(vm_minus14, tr14, out=np.zeros_like(vm_minus14), where=tr14!=0)
    
    # Chopiness Index (using 14-period)
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr14 / (highest_high14 - lowest_low14)) / np.log10(14)
    
    # Volume confirmation (20-period MA)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or 
            np.isnan(chop[i]) or np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Chop filter: CHOP > 61.8 indicates ranging market (good for mean reversion)
        chop_filter = chop[i] > 61.8
        
        # Entry conditions
        if position == 0:
            # Long: VI+ > VI- + volume + chop
            if (vi_plus[i] > vi_minus[i] and 
                volume_filter and chop_filter):
                signals[i] = 0.25
                position = 1
            # Short: VI- > VI+ + volume + chop
            elif (vi_minus[i] > vi_plus[i] and 
                  volume_filter and chop_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: VI- crosses above VI+ or filters fail
            if vi_minus[i] > vi_plus[i] or not volume_filter or not chop_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: VI+ crosses above VI- or filters fail
            if vi_plus[i] > vi_minus[i] or not volume_filter or not chop_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Vortex_Trend_Plus_Volume_Regime"
timeframe = "4h"
leverage = 1.0
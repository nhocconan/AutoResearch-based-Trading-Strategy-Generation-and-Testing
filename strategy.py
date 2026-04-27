#!/usr/bin/env python3
"""
#100786 - 4h_Vortex_Trend_Volume_Spike
Hypothesis: Trend-following strategy using Vortex Indicator (VI+) and (VI-) to detect trend direction, combined with volume spikes for confirmation.
Works in bull markets (strong VI+ > VI- with volume) and bear markets (strong VI- > VI+ with volume).
Uses 4h primary timeframe with 1d HTF for Vortex calculation to reduce noise and avoid false signals.
Targets 20-50 trades/year to minimize fee drag.
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
    
    # Get 1d data for Vortex Indicator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for Vortex calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value has no previous close
    
    # Vortex Indicator components
    vm_plus = np.abs(high_1d - np.roll(low_1d, 1))
    vm_minus = np.abs(low_1d - np.roll(high_1d, 1))
    vm_plus[0] = 0
    vm_minus[0] = 0
    
    # Sum over 14 periods (standard Vortex period)
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vm_plus14 = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus14 = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Vortex Indicator lines
    vi_plus = vm_plus14 / tr14
    vi_minus = vm_minus14 / tr14
    
    # Align VI lines to 4h timeframe
    vi_plus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_minus)
    
    # Volume filter: volume > 1.8x 20-period average (higher threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vi_plus_aligned[i]) or np.isnan(vi_minus_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: VI+ > VI- (bullish trend) + volume spike
        if (vi_plus_aligned[i] > vi_minus_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: VI- > VI+ (bearish trend) + volume spike
        elif (vi_minus_aligned[i] > vi_plus_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend weakening (VI lines cross in opposite direction)
        elif position == 1 and vi_minus_aligned[i] > vi_plus_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and vi_plus_aligned[i] > vi_minus_aligned[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Vortex_Trend_Volume_Spike"
timeframe = "4h"
leverage = 1.0
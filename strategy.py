# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
12h_Vortex_Trend_With_Volume_Filter
Hypothesis: Trade Vortex trend direction with volume confirmation on 12h timeframe.
Long when VI+ > VI- and volume > 1.5x 20-period average volume; short when VI- > VI+ and volume > 1.5x average.
Exit when Vortex signal reverses.
Vortex indicator identifies trend direction and strength, less prone to whipsaw than simple MA crossovers.
Volume filter ensures trades occur during active market phases, reducing false signals in low-volume periods.
Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.
Works in bull/bear: volume filter avoids low-activity periods, Vortex adapts to trend changes.
"""

name = "12h_Vortex_Trend_With_Volume_Filter"
timeframe = "12h"
leverage = 1.0

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
    
    # Calculate Vortex Indicator (VI)
    # VM+ = |high - low_previous|, VM- = |low - high_previous|
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = 0  # first element has no previous
    vm_minus[0] = 0
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first element
    
    # Period for Vortex (typically 14)
    period = 14
    
    # Sum of VM+, VM-, and TR over period
    sum_vm_plus = np.zeros(n)
    sum_vm_minus = np.zeros(n)
    sum_tr = np.zeros(n)
    
    for i in range(period, n):
        sum_vm_plus[i] = np.sum(vm_plus[i-period+1:i+1])
        sum_vm_minus[i] = np.sum(vm_minus[i-period+1:i+1])
        sum_tr[i] = np.sum(tr[i-period+1:i+1])
    
    # VI+ and VI-
    vi_plus = np.zeros(n)
    vi_minus = np.zeros(n)
    for i in range(period, n):
        if sum_tr[i] > 0:
            vi_plus[i] = sum_vm_plus[i] / sum_tr[i]
            vi_minus[i] = sum_vm_minus[i] / sum_tr[i]
        else:
            vi_plus[i] = 0
            vi_minus[i] = 0
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    volume_filter = np.zeros(n, dtype=bool)
    for i in range(20, n):
        if vol_ma[i] > 0:
            volume_filter[i] = volume[i] > 1.5 * vol_ma[i]
        else:
            volume_filter[i] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, period)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or invalid
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or 
            np.isnan(close[i]) or not volume_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: VI+ > VI- and volume filter
            if vi_plus[i] > vi_minus[i]:
                signals[i] = 0.25
                position = 1
            # Short: VI- > VI+ and volume filter
            elif vi_minus[i] > vi_plus[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: VI- > VI+ (trend reversal)
            if vi_minus[i] > vi_plus[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: VI+ > VI- (trend reversal)
            if vi_plus[i] > vi_minus[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
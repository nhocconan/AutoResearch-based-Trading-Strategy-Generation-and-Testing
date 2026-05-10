#!/usr/bin/env python3
# 4h_Vortex_Trend_Filter
# Hypothesis: Uses Vortex Indicator (VI+) and VI- to determine trend direction from 1d timeframe.
# Enters long when VI+ crosses above VI- in an uptrend with volume confirmation.
# Enters short when VI- crosses above VI+ in a downtrend with volume confirmation.
# Uses Vortex on daily timeframe for trend, with 4h price action for entry timing.
# Designed for low trade frequency (<30/year) to avoid fee drag, with trend-following bias.

name = "4h_Vortex_Trend_Filter"
timeframe = "4h"
leverage = 1.0

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
    
    # 1d data for Vortex trend direction
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Vortex Indicator (VI+ and VI-)
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # VM+ and VM-
    vm_plus = np.abs(high_1d - np.roll(low_1d, 1))
    vm_minus = np.abs(low_1d - np.roll(high_1d, 1))
    vm_plus[0] = 0
    vm_minus[0] = 0
    
    # Sum over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vm_plus_sum = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    
    # VI+ and VI-
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum
    
    # Align Vortex to 4h timeframe
    vi_plus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_minus)
    
    # Volume confirmation: current volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(vi_plus_aligned[i]) or np.isnan(vi_minus_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: VI+ crosses above VI- with volume confirmation
            if (vi_plus_aligned[i] > vi_minus_aligned[i] and 
                vi_plus_aligned[i-1] <= vi_minus_aligned[i-1] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: VI- crosses above VI+ with volume confirmation
            elif (vi_minus_aligned[i] > vi_plus_aligned[i] and 
                  vi_minus_aligned[i-1] <= vi_plus_aligned[i-1] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: VI- crosses above VI+ (trend reversal)
            if vi_minus_aligned[i] > vi_plus_aligned[i] and vi_minus_aligned[i-1] <= vi_plus_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: VI+ crosses above VI- (trend reversal)
            if vi_plus_aligned[i] > vi_minus_aligned[i] and vi_plus_aligned[i-1] <= vi_minus_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
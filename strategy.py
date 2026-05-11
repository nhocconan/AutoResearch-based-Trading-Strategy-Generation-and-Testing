#!/usr/bin/env python3
name = "4h_HTF_1w_Vortex_VolumeSpike"
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
    
    # Weekly Vortex indicator (VI+ and VI-)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # VM+ and VM-
    vm_plus = np.abs(high_1w - np.roll(low_1w, 1))
    vm_minus = np.abs(low_1w - np.roll(high_1w, 1))
    vm_plus[0] = np.nan
    vm_minus[0] = np.nan
    
    # Sum over 14 periods
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vm_plus14 = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus14 = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    
    vi_plus = vm_plus14 / tr14
    vi_minus = vm_minus14 / tr14
    
    # Align to 4h
    vi_plus_aligned = align_htf_to_ltf(prices, df_1w, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1w, vi_minus)
    
    # Volume confirmation (4h volume > 2.0x 20-period average)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 2.0 * volume_ma20
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        if np.isnan(vi_plus_aligned[i]) or np.isnan(vi_minus_aligned[i]) or np.isnan(volume_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: VI+ > VI- (bullish trend) + volume spike
            if vi_plus_aligned[i] > vi_minus_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: VI- > VI+ (bearish trend) + volume spike
            elif vi_minus_aligned[i] > vi_plus_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: VI- crosses above VI+ (trend reversal)
            if vi_minus_aligned[i] > vi_plus_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: VI+ crosses above VI- (trend reversal)
            if vi_plus_aligned[i] > vi_minus_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
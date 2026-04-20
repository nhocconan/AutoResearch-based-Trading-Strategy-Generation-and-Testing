#!/usr/bin/env python3
# 4h_1d_Vortex_TrendFollowing_VolumeConfirmation
# Hypothesis: Use daily Vortex indicator (VI+ and VI-) to capture trend direction on 4h timeframe.
# Vortex identifies trend strength by comparing current high-low ranges with prior periods.
# Long when VI+ > VI- (uptrend) with volume confirmation; short when VI- > VI+ (downtrend) with volume confirmation.
# Uses volume spike (>1.5x 24-period average) to filter false signals and reduce whipsaw.
# Target: 20-40 trades/year per symbol for balance between signal quality and frequency.

name = "4h_1d_Vortex_TrendFollowing_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 14-period Vortex Indicator on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range components
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First element NaN
    
    # Vortex Indicator components
    vm_plus = np.abs(high_1d[1:] - low_1d[:-1])  # |High - Prior Low|
    vm_minus = np.abs(low_1d[1:] - high_1d[:-1])  # |Low - Prior High|
    vm_plus = np.concatenate([[np.nan], vm_plus])
    vm_minus = np.concatenate([[np.nan], vm_minus])
    
    # Sum over 14 periods
    tr14 = np.convolve(tr, np.ones(14), 'valid') if len(tr) >= 14 else np.full_like(tr, np.nan)
    vm_plus14 = np.convolve(vm_plus, np.ones(14), 'valid') if len(vm_plus) >= 14 else np.full_like(vm_plus, np.nan)
    vm_minus14 = np.convolve(vm_minus, np.ones(14), 'valid') if len(vm_minus) >= 14 else np.full_like(vm_minus, np.nan)
    
    # Align arrays to original length
    tr14_full = np.full_like(tr, np.nan)
    vm_plus14_full = np.full_like(vm_plus, np.nan)
    vm_minus14_full = np.full_like(vm_minus, np.nan)
    if len(tr14) > 0:
        tr14_full[13:13+len(tr14)] = tr14
        vm_plus14_full[13:13+len(vm_plus14)] = vm_plus14
        vm_minus14_full[13:13+len(vm_minus14)] = vm_minus14
    
    # VI+ and VI- (avoid division by zero)
    vi_plus = np.where(tr14_full != 0, vm_plus14_full / tr14_full, 0)
    vi_minus = np.where(tr14_full != 0, vm_minus14_full / tr14_full, 0)
    
    # Calculate volume average for spike detection
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # 24*4h = 4 days
    
    # Align 1d indicators to 4h timeframe
    vi_plus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_minus)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vi_plus_aligned[i]) or np.isnan(vi_minus_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5 * 1d average volume
        volume_spike = volume[i] > 1.5 * vol_ma_aligned[i]
        
        if position == 0:
            # Long: VI+ > VI- (uptrend) with volume confirmation
            if vi_plus_aligned[i] > vi_minus_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: VI- > VI+ (downtrend) with volume confirmation
            elif vi_minus_aligned[i] > vi_plus_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if trend reverses (VI- > VI+)
            if vi_minus_aligned[i] > vi_plus_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if trend reverses (VI+ > VI-)
            if vi_plus_aligned[i] > vi_minus_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
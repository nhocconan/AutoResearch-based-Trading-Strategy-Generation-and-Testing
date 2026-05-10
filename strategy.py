#!/usr/bin/env python3
"""
4h_Vortex_Trend_Filter_Volume
Hypothesis: Vortex indicator identifies trend direction by comparing upward and downward movement.
Combined with volume confirmation (current volume > 1.5x average volume) to filter for high-conviction moves.
Works in both bull and bear markets by following established trends with volume confirmation.
Targets 20-40 trades/year by requiring trend alignment and volume expansion.
"""

name = "4h_Vortex_Trend_Filter_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for trend and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Vortex Indicator (VI) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(np.diff(high_1d))
    tr2 = np.abs(np.diff(low_1d))
    tr3 = np.abs(np.diff(high_1d, low_1d))  # |high - low|
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Prepend first value for alignment
    tr = np.concatenate([[high_1d[0] - low_1d[0]], tr])
    
    # Vortex movements
    vm_plus = np.abs(np.diff(high_1d, np.roll(low_1d, 1)))  # |high - low_prev|
    vm_minus = np.abs(np.diff(low_1d, np.roll(high_1d, 1)))  # |low - high_prev|
    # Prepend first values
    vm_plus = np.concatenate([[np.abs(high_1d[0] - low_1d[0])], vm_plus])
    vm_minus = np.concatenate([[np.abs(low_1d[0] - high_1d[0])], vm_minus])
    
    # Smooth over 14 periods (standard Vortex period)
    period = 14
    tr_sum = np.zeros_like(tr)
    vm_plus_sum = np.zeros_like(vm_plus)
    vm_minus_sum = np.zeros_like(vm_minus)
    
    for i in range(len(tr)):
        if i < period:
            tr_sum[i] = np.sum(tr[:i+1])
            vm_plus_sum[i] = np.sum(vm_plus[:i+1])
            vm_minus_sum[i] = np.sum(vm_minus[:i+1])
        else:
            tr_sum[i] = tr_sum[i-1] - tr_sum[i-period] + tr[i]
            vm_plus_sum[i] = vm_plus_sum[i-1] - vm_plus_sum[i-period] + vm_plus[i]
            vm_minus_sum[i] = vm_minus_sum[i-1] - vm_minus_sum[i-period] + vm_minus[i]
    
    # Avoid division by zero
    vi_plus = np.where(tr_sum > 0, vm_plus_sum / tr_sum, 0)
    vi_minus = np.where(tr_sum > 0, vm_minus_sum / tr_sum, 0)
    
    # Align VI to 4h
    vi_plus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_minus)
    
    # Calculate 1d average volume for volume filter
    vol_avg_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need VI (14) and volume average (20)
    start_idx = max(14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(vi_plus_aligned[i]) or 
            np.isnan(vi_minus_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Higher timeframe trend filter (1d): VI+ > VI- indicates uptrend
        uptrend_1d = vi_plus_aligned[i] > vi_minus_aligned[i]
        downtrend_1d = vi_plus_aligned[i] < vi_minus_aligned[i]
        
        # Volume filter: current 4h volume > 1.5x average 1d volume (scaled)
        vol_4h = volume[i]
        # Scale 1d volume to 4h equivalent (1d = 6x 4h)
        vol_4h_equiv = vol_avg_1d_aligned[i] / 6.0
        volume_filter = vol_4h > vol_4h_equiv * 1.5
        
        if position == 0:
            # Long entry: VI+ > VI- (uptrend) + volume participation
            if uptrend_1d and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: VI- > VI+ (downtrend) + volume participation
            elif downtrend_1d and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend changes to downtrend or volume dries up
            if not uptrend_1d:  # trend changed
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend changes to uptrend or volume dries up
            if not downtrend_1d:  # trend changed
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
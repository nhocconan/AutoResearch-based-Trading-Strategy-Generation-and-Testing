#!/usr/bin/env python3
# 12h_Vortex_Trend_With_Volume_Filter
# Hypothesis: Vortex Indicator identifies trend direction (VI+ > VI- for uptrend, VI- > VI+ for downtrend).
# Combined with volume confirmation (current volume > 1.5x 24-period average) to filter weak breakouts.
# Works in bull markets (riding uptrends) and bear markets (riding downtrends) by following the trend.
# Low trade frequency expected due to trend filter + volume confirmation.

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
    
    # 1d data for Vortex calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for Vortex
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Vortex Indicator components
    vm_plus = np.abs(high_1d - np.roll(low_1d, 1))
    vm_minus = np.abs(low_1d - np.roll(high_1d, 1))
    vm_plus[0] = np.abs(high_1d[0] - low_1d[0])
    vm_minus[0] = np.abs(low_1d[0] - high_1d[0])
    
    # Vortex Indicator (14-period)
    period = 14
    def sum_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.sum(arr[i-p+1:i+1])
        return res
    
    tr_sum = sum_arr(tr, period)
    vm_plus_sum = sum_arr(vm_plus, period)
    vm_minus_sum = sum_arr(vm_minus, period)
    
    vi_plus = np.where(tr_sum > 0, vm_plus_sum / tr_sum, 0)
    vi_minus = np.where(tr_sum > 0, vm_minus_sum / tr_sum, 0)
    
    # Align Vortex to 12h timeframe (wait for 1d bar to close)
    vi_plus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_minus)
    
    # Volume confirmation (24-period average = 12 days for 12h timeframe)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, period)  # Need enough history for Vortex
    
    for i in range(start_idx, n):
        if np.isnan(vi_plus_aligned[i]) or np.isnan(vi_minus_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: VI+ > VI- (uptrend) + volume confirmation
            if vi_plus_aligned[i] > vi_minus_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: VI- > VI+ (downtrend) + volume confirmation
            elif vi_minus_aligned[i] > vi_plus_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend reverses (VI- > VI+)
            if vi_minus_aligned[i] > vi_plus_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend reverses (VI+ > VI-)
            if vi_plus_aligned[i] > vi_minus_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
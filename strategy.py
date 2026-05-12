#!/usr/bin/env python3
name = "4h_Vortex_Signal_1dTrend_VolumeFilter"
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
    
    # 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Vortex Indicator on 4h
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = high[0] - low[0]
    vm_minus[0] = high[0] - low[0]
    
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vm_plus_sum = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum
    
    # Volume filter: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14, 20)  # ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if 1d trend data not ready
        if np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: VI+ crosses above VI- + 1d uptrend + volume filter
            if (vi_plus[i] > vi_minus[i] and vi_plus[i-1] <= vi_minus[i-1] and  # crossover
                close[i] > ema50_1d_aligned[i] and                             # 1d uptrend
                volume_filter[i]):                                             # volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: VI- crosses above VI+ + 1d downtrend + volume filter
            elif (vi_minus[i] > vi_plus[i] and vi_minus[i-1] <= vi_plus[i-1] and  # crossover
                  close[i] < ema50_1d_aligned[i] and                             # 1d downtrend
                  volume_filter[i]):                                             # volume confirmation
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when VI- crosses above VI+ or 1d trend flips
            if (vi_minus[i] > vi_plus[i] or close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when VI+ crosses above VI- or 1d trend flips
            if (vi_plus[i] > vi_minus[i] or close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
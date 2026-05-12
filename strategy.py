#!/usr/bin/env python3
name = "4h_Vortex_Trend_1dATR_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Vortex Indicator ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Vortex Movement
    vm_plus = np.abs(high_1d - np.roll(low_1d, 1))
    vm_minus = np.abs(low_1d - np.roll(high_1d, 1))
    vm_plus[0] = np.abs(high_1d[0] - low_1d[0])
    vm_minus[0] = np.abs(high_1d[0] - low_1d[0])
    
    # Sum over 14 periods
    period = 14
    tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    vm_plus_sum = pd.Series(vm_plus).rolling(window=period, min_periods=period).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=period, min_periods=period).sum().values
    
    # Vortex Indicator
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum
    
    # === 1d ATR for volatility filter ===
    atr = tr_sum / period
    atr_ma = pd.Series(atr).rolling(window=10, min_periods=10).mean().values
    atr_filter = atr > atr_ma  # Volatility expansion
    
    # === 1d Volume spike filter ===
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (2.0 * vol_avg_1d)
    
    # Align all indicators
    vi_plus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_minus)
    atr_filter_aligned = align_htf_to_ltf(prices, df_1d, atr_filter.astype(float))
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vi_plus_aligned[i]) or 
            np.isnan(vi_minus_aligned[i]) or
            np.isnan(atr_filter_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: VI+ > VI- (bullish trend) + volatility expansion + volume spike
            if (vi_plus_aligned[i] > vi_minus_aligned[i] and
                atr_filter_aligned[i] > 0.5 and
                vol_spike_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: VI- > VI+ (bearish trend) + volatility expansion + volume spike
            elif (vi_minus_aligned[i] > vi_plus_aligned[i] and
                  atr_filter_aligned[i] > 0.5 and
                  vol_spike_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: VI- > VI+ (trend reversal) or volatility contraction
            if vi_minus_aligned[i] > vi_plus_aligned[i] or atr_filter_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: VI+ > VI- (trend reversal) or volatility contraction
            if vi_plus_aligned[i] > vi_minus_aligned[i] or atr_filter_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
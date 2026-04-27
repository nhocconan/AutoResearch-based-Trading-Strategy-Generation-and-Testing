#!/usr/bin/env python3
"""
12h_Vortex_Trend_1wTrend_VolumeSpike
Hypothesis: Vortex indicator (VI+) on 12h combined with weekly trend filter and volume spike captures strong trend moves while avoiding whipsaws in ranging markets. Works in both bull and bear markets by only taking trades aligned with higher timeframe trend. Target: 15-25 trades/year per symbol.
"""

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
    
    # Get 1w data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA200 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema200_1w = close_1w.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Get 12h data for Vortex indicator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range calculation
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original array
    
    # Vortex movement
    vm_plus = np.abs(high_12h[1:] - low_12h[:-1])
    vm_minus = np.abs(low_12h[1:] - high_12h[:-1])
    vm_plus = np.concatenate([[np.nan], vm_plus])
    vm_minus = np.concatenate([[np.nan], vm_minus])
    
    # Sum over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vm_plus_sum = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Vortex indicators
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum
    
    # Align to 12h timeframe (Vortex already calculated on 12h)
    vi_plus_aligned = align_htf_to_ltf(prices, df_12h, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_12h, vi_minus)
    
    # Volume spike detection (20-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period - need enough data for all indicators
    start_idx = max(100, 20)  # 100 for weekly EMA200 warmup, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(vi_plus_aligned[i]) or 
            np.isnan(vi_minus_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: VI+ > VI- (bullish vortex) AND price above weekly EMA200 AND volume spike
            if (vi_plus_aligned[i] > vi_minus_aligned[i] and 
                close[i] > ema200_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: VI- > VI+ (bearish vortex) AND price below weekly EMA200 AND volume spike
            elif (vi_minus_aligned[i] > vi_plus_aligned[i] and 
                  close[i] < ema200_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Vortex turns bearish OR price crosses below weekly EMA200
            if (vi_minus_aligned[i] > vi_plus_aligned[i] or 
                close[i] < ema200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Vortex turns bullish OR price crosses above weekly EMA200
            if (vi_plus_aligned[i] > vi_minus_aligned[i] or 
                close[i] > ema200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Vortex_Trend_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0
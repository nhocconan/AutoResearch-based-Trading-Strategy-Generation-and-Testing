#!/usr/bin/env python3
# 4h_1dVortex_VIplus_Minus_Volume
# Uses daily Vortex indicator (VI+ and VI-) to detect trend direction with 4h volume confirmation.
# Long when VI+ > VI- and price above prior close, short when VI- > VI+ and price below prior close.
# Vortex helps distinguish true trends from whipsaws, effective in both bull and bear markets.
# Target: 75-200 total trades over 4 years with 0.25 position sizing.

name = "4h_1dVortex_VIplus_Minus_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Vortex indicator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # Vortex Indicator components
    vm_plus = np.abs(high_1d - np.roll(low_1d, 1))
    vm_minus = np.abs(low_1d - np.roll(high_1d, 1))
    vm_plus[0] = 0  # No previous period for first bar
    vm_minus[0] = 0
    
    # Sum over 14 periods (standard Vortex period)
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vm_plus_sum = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    vi_plus = np.where(tr_sum != 0, vm_plus_sum / tr_sum, 0)
    vi_minus = np.where(tr_sum != 0, vm_minus_sum / tr_sum, 0)
    
    # Align Vortex values to 4h timeframe
    vi_plus_4h = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_4h = align_htf_to_ltf(prices, df_1d, vi_minus)
    
    # 4h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(vi_plus_4h[i]) or np.isnan(vi_minus_4h[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: VI+ > VI- (bullish trend) with volume confirmation
            if vi_plus_4h[i] > vi_minus_4h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: VI- > VI+ (bearish trend) with volume confirmation
            elif vi_minus_4h[i] > vi_plus_4h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: trend weakness (VI- crosses above VI+) or volume drops
            if vi_minus_4h[i] > vi_plus_4h[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend weakness (VI+ crosses above VI-) or volume drops
            if vi_plus_4h[i] > vi_minus_4h[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf
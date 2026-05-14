#!/usr/bin/env python3
# 4h_Vortex_Trend_Filter_v2
# Hypothesis: Reduced-trading version of Vortex strategy. Uses Vortex(14) from daily timeframe
# combined with 4h EMA50 filter and volume confirmation to reduce false signals.
# Only enters when Vortex shows strong trend (VI+ > VI- by threshold) and price is
# aligned with EMA50. Volume must be above average to confirm conviction.
# Targets 20-40 trades/year to avoid fee drag while maintaining trend-following edge.

name = "4h_Vortex_Trend_Filter_v2"
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
    
    # Get 1d data for Vortex calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Vortex Indicator components (VM+ and VM-)
    vm_plus = np.abs(high_1d[1:] - low_1d[:-1])
    vm_minus = np.abs(low_1d[1:] - high_1d[:-1])
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum over 14 periods
    n_1d = len(high_1d)
    vi_plus = np.zeros(n_1d)
    vi_minus = np.zeros(n_1d)
    
    for i in range(14, n_1d):
        if i >= 14:
            sum_vm_plus = np.sum(vm_plus[i-13:i+1])
            sum_vm_minus = np.sum(vm_minus[i-13:i+1])
            sum_tr = np.sum(tr[i-13:i+1])
            if sum_tr > 0:
                vi_plus[i] = sum_vm_plus / sum_tr
                vi_minus[i] = sum_vm_minus / sum_tr
    
    # Align Vortex indicators to 4h timeframe
    vi_plus_4h = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_4h = align_htf_to_ltf(prices, df_1d, vi_minus)
    
    # 4h EMA50 for stronger trend filter
    ema_50_4h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(vi_plus_4h[i]) or np.isnan(vi_minus_4h[i]) or 
            np.isnan(ema_50_4h[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Strong trend confirmation: VI+ > VI- by 0.02 threshold
            vol_ok = volume[i] > vol_ma[i]
            
            # Long: strong bullish vortex + price above EMA50 + volume confirmation
            if (vi_plus_4h[i] > vi_minus_4h[i] + 0.02 and 
                close[i] > ema_50_4h[i] and vol_ok):
                signals[i] = 0.25
                position = 1
            # Short: strong bearish vortex + price below EMA50 + volume confirmation
            elif (vi_minus_4h[i] > vi_plus_4h[i] + 0.02 and 
                  close[i] < ema_50_4h[i] and vol_ok):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: trend weakens or reverses
            if (vi_minus_4h[i] >= vi_plus_4h[i] or 
                close[i] < ema_50_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: trend weakens or reverses
            if (vi_plus_4h[i] >= vi_minus_4h[i] or 
                close[i] > ema_50_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
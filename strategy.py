#!/usr/bin/env python3
# 4h_Vortex_Vortex_Trend_Filter
# Hypothesis: The Vortex Indicator (VI) identifies trend strength and direction. 
# When VI+ > VI- with confirmation from a higher timeframe trend (1d EMA34) and volume spike, 
# it signals a strong uptrend. Conversely, VI- > VI+ with downtrend confirmation signals a downtrend.
# This strategy captures strong trending moves while avoiding choppy markets by requiring 
# volume confirmation and higher timeframe alignment. Designed to work in both bull and bear markets
# by following the primary trend on the daily timeframe.
# Timeframe: 4h, uses 1d trend filter for multi-timeframe alignment.
# Low trade frequency (~20-30/year) via strict Vortex crossover + volume + trend confluence.

timeframe = "4h"
name = "4h_Vortex_Vortex_Trend_Filter"
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
    
    # Calculate Vortex Indicator (VI) on 4h data
    # VI+ and VI- over period 14 (standard)
    period = 14
    tr0 = np.maximum(high, np.roll(high, 1))
    tr1 = np.minimum(low, np.roll(low, 1))
    tr = np.maximum(tr0 - tr1, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr = np.where(np.isnan(tr), 0, tr)  # handle first bar
    
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    
    # Sum over period
    tr_sum = np.convolve(tr, np.ones(period), 'full')[:n]
    vm_plus_sum = np.convolve(vm_plus, np.ones(period), 'full')[:n]
    vm_minus_sum = np.convolve(vm_minus, np.ones(period), 'full')[:n]
    
    # Adjust for convolution offset
    tr_sum = tr_sum[period-1:period-1+n]
    vm_plus_sum = vm_plus_sum[period-1:period-1+n]
    vm_minus_sum = vm_minus_sum[period-1:period-1+n]
    
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum
    
    # Handle division by zero
    vi_plus = np.where(tr_sum == 0, 0, vi_plus)
    vi_minus = np.where(tr_sum == 0, 0, vi_minus)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Daily trend filter: EMA34
    ema_34_1d = pd.Series(d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 2x average volume (24-period = 1 day on 4h chart)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period, 24, 34)  # Ensure we have VI, volume MA and EMA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN or invalid
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: VI+ > VI- with volume spike and daily uptrend
            if vi_plus[i] > vi_minus[i] and volume[i] > 2.0 * vol_ma[i] and close[i] > ema_34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: VI- > VI+ with volume spike and daily downtrend
            elif vi_minus[i] > vi_plus[i] and volume[i] > 2.0 * vol_ma[i] and close[i] < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: VI crossover (VI- > VI+) or trend failure
            if vi_minus[i] > vi_plus[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: VI crossover (VI+ > VI-) or trend failure
            if vi_plus[i] > vi_minus[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
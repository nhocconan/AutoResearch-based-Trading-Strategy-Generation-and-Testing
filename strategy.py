#!/usr/bin/env python3
# 4h_Vortex_Vortex_Trend_Filter
# Hypothesis: Vortex indicator (VI+ and VI-) identifies trend direction, with VI+ > VI- indicating uptrend and VI- > VI+ indicating downtrend.
# Combined with Vortex crossovers for entry/exit and volume confirmation to filter false signals.
# Works in bull/bear: Trend filter prevents counter-trend trades, volume ensures institutional participation.
# Uses 1-day Vortex for trend alignment and 4-hour for entry timing.

name = "4h_Vortex_Vortex_Trend_Filter"
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
    
    # Calculate 1-day Vortex for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d = np.concatenate([[np.nan], tr_1d])  # Align with index
    
    # Vortex Indicator components
    vm_plus = np.abs(high_1d[1:] - low_1d[:-1])
    vm_minus = np.abs(low_1d[1:] - high_1d[:-1])
    vm_plus = np.concatenate([[np.nan], vm_plus])
    vm_minus = np.concatenate([[np.nan], vm_minus])
    
    # Sum over 14 periods (standard Vortex period)
    tr14_1d = np.full_like(tr_1d, np.nan)
    vm_plus14_1d = np.full_like(vm_plus, np.nan)
    vm_minus14_1d = np.full_like(vm_minus, np.nan)
    
    if len(tr_1d) >= 14:
        for i in range(14, len(tr_1d)):
            tr14_1d[i] = np.nansum(tr_1d[i-13:i+1])
            vm_plus14_1d[i] = np.nansum(vm_plus[i-13:i+1])
            vm_minus14_1d[i] = np.nansum(vm_minus[i-13:i+1])
    
    # VI+ and VI-
    vi_plus_1d = np.full_like(tr_1d, np.nan)
    vi_minus_1d = np.full_like(tr_1d, np.nan)
    valid = (~np.isnan(tr14_1d)) & (tr14_1d != 0)
    vi_plus_1d[valid] = vm_plus14_1d[valid] / tr14_1d[valid]
    vi_minus_1d[valid] = vm_minus14_1d[valid] / tr14_1d[valid]
    
    # Trend: VI+ > VI- indicates uptrend, VI- > VI+ indicates downtrend
    vi_plus_1d_aligned = align_htf_to_ltf(prices, df_1d, vi_plus_1d)
    vi_minus_1d_aligned = align_htf_to_ltf(prices, df_1d, vi_minus_1d)
    
    # Calculate 4-hour Vortex for entry signals
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    vm_plus = np.abs(high[1:] - low[:-1])
    vm_minus = np.abs(low[1:] - high[:-1])
    vm_plus = np.concatenate([[np.nan], vm_plus])
    vm_minus = np.concatenate([[np.nan], vm_minus])
    
    # Sum over 14 periods
    tr14 = np.full_like(tr, np.nan)
    vm_plus14 = np.full_like(vm_plus, np.nan)
    vm_minus14 = np.full_like(vm_minus, np.nan)
    
    if len(tr) >= 14:
        for i in range(14, len(tr)):
            tr14[i] = np.nansum(tr[i-13:i+1])
            vm_plus14[i] = np.nansum(vm_plus[i-13:i+1])
            vm_minus14[i] = np.nansum(vm_minus[i-13:i+1])
    
    # VI+ and VI-
    vi_plus = np.full_like(tr, np.nan)
    vi_minus = np.full_like(tr, np.nan)
    valid = (~np.isnan(tr14)) & (tr14 != 0)
    vi_plus[valid] = vm_plus14[valid] / tr14[valid]
    vi_minus[valid] = vm_minus14[valid] / tr14[valid]
    
    # Vortex crossover signals: VI+ crosses above VI- = buy, VI- crosses above VI+ = sell
    vi_plus_cross_vi_minus = np.full_like(vi_plus, False)
    vi_minus_cross_vi_plus = np.full_like(vi_minus, False)
    
    for i in range(1, len(vi_plus)):
        if not np.isnan(vi_plus[i]) and not np.isnan(vi_minus[i]) and not np.isnan(vi_plus[i-1]) and not np.isnan(vi_minus[i-1]):
            vi_plus_cross_vi_minus[i] = (vi_plus[i] > vi_minus[i]) and (vi_plus[i-1] <= vi_minus[i-1])
            vi_minus_cross_vi_plus[i] = (vi_minus[i] > vi_plus[i]) and (vi_minus[i-1] <= vi_plus[i-1])
    
    # Volume confirmation: current volume / 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(20, len(volume)):
            vol_ma[i] = np.nanmean(volume[i-19:i+1])
        # Initialize first 19 values with NaN, 20th with mean of first 20
        if len(volume) >= 20:
            vol_ma[19] = np.nanmean(volume[0:20])
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Ensure volume MA and Vortex are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vi_plus_1d_aligned[i]) or np.isnan(vi_minus_1d_aligned[i]) or
            np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: VI+ crosses above VI- (bullish) AND uptrend (VI+ > VI- on 1d) AND volume spike
            if (vi_plus_cross_vi_minus[i] and 
                vi_plus_1d_aligned[i] > vi_minus_1d_aligned[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: VI- crosses above VI+ (bearish) AND downtrend (VI- > VI+ on 1d) AND volume spike
            elif (vi_minus_cross_vi_plus[i] and 
                  vi_minus_1d_aligned[i] > vi_plus_1d_aligned[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: VI- crosses above VI+ (bearish crossover) OR trend reversal (VI- > VI+ on 1d)
            if vi_minus_cross_vi_plus[i] or (vi_minus_1d_aligned[i] > vi_plus_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: VI+ crosses above VI- (bullish crossover) OR trend reversal (VI+ > VI- on 1d)
            if vi_plus_cross_vi_minus[i] or (vi_plus_1d_aligned[i] > vi_minus_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
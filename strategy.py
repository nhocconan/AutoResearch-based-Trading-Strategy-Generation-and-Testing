#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Vortex_Trend_Volume_Confirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h data for price and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Vortex indicator (HTF)
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 2:
        return np.zeros(n)
    
    # Calculate Vortex indicator on daily data
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    # True Range
    tr1 = np.abs(high_d[1:] - low_d[1:])
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Vortex Indicator components
    vm_plus = np.abs(high_d[1:] - low_d[:-1])  # |High - Prev Low|
    vm_minus = np.abs(low_d[1:] - high_d[:-1])  # |Low - Prev High|
    vm_plus = np.concatenate([[np.nan], vm_plus])
    vm_minus = np.concatenate([[np.nan], vm_minus])
    
    # Sum over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vm_plus_sum = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Vortex values
    vi_plus = vm_plus_sum / (tr_sum + 1e-10)
    vi_minus = vm_minus_sum / (tr_sum + 1e-10)
    
    # Align Vortex to 4h timeframe
    vi_plus_aligned = align_htf_to_ltf(prices, df_d, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_d, vi_minus)
    
    # Volume filter on 4h: volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Need enough data for volume MA and Vortex
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(vi_plus_aligned[i]) or 
            np.isnan(vi_minus_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vi_plus_val = vi_plus_aligned[i]
        vi_minus_val = vi_minus_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: VI+ > VI- + volume filter
            if vi_plus_val > vi_minus_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: VI- > VI+ + volume filter
            elif vi_minus_val > vi_plus_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: VI- crosses above VI+
            if vi_minus_val > vi_plus_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: VI+ crosses above VI-
            if vi_plus_val > vi_minus_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
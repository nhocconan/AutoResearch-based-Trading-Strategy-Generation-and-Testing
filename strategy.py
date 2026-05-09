#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Vortex_Trend_Volume"
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
    
    # Get 1d data for Vortex indicator and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Vortex Indicator on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Vortex components
    vm_plus = np.abs(high_1d[1:] - low_1d[:-1])
    vm_minus = np.abs(low_1d[1:] - high_1d[:-1])
    vm_plus = np.concatenate([[np.nan], vm_plus])
    vm_minus = np.concatenate([[np.nan], vm_minus])
    
    # Sum over 34 periods
    tr_sum = pd.Series(tr).rolling(window=34, min_periods=34).sum().values
    vm_plus_sum = pd.Series(vm_plus).rolling(window=34, min_periods=34).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=34, min_periods=34).sum().values
    
    # Vortex indicators
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum
    
    # Align Vortex indicators to 4h timeframe
    vi_plus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_minus)
    
    # Calculate 1d volume average for spike detection
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Need 34 for Vortex and 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(vi_plus_aligned[i]) or np.isnan(vi_minus_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vi_plus_val = vi_plus_aligned[i]
        vi_minus_val = vi_minus_aligned[i]
        vol_ma_val = vol_ma_1d_aligned[i]
        vol = volume[i]
        
        if position == 0:
            # Enter long: VI+ > VI- (bullish trend) with volume confirmation
            if vi_plus_val > vi_minus_val and vol > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Enter short: VI- > VI+ (bearish trend) with volume confirmation
            elif vi_minus_val > vi_plus_val and vol > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Trend weakness (VI- > VI+) or volume drops
            if vi_minus_val > vi_plus_val or vol < vol_ma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Trend weakness (VI+ > VI-) or volume drops
            if vi_plus_val > vi_minus_val or vol < vol_ma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
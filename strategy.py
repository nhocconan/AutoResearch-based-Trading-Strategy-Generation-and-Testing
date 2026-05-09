#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Vortex_Trend_With_Volume_And_Chop_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Vortex and Chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Vortex Indicator on 1d
    # VM = |High - Prev Close|, VM = |Low - Prev Close|
    vm_plus = np.abs(high[1:] - close[:-1])
    vm_minus = np.abs(low[1:] - close[:-1])
    tr = np.maximum(
        np.abs(high[1:] - low[1:]),
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
    )
    # Pad first value with 0 for alignment
    vm_plus = np.concatenate([[0], vm_plus])
    vm_minus = np.concatenate([[0], vm_minus])
    tr = np.concatenate([[0], tr])
    
    # Sum over 14 periods
    vi_plus = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vi_minus = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    vt_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Vortex lines
    vi_plus_norm = vi_plus / vt_tr
    vi_minus_norm = vi_minus / vt_tr
    
    # Chop Index on 1d: measures sideways vs trending
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10((highest_high - lowest_low) / np.sum(tr[-14:]) if len(tr) >= 14 else np.nan) / np.log10(14)
    chop = pd.Series(chop).rolling(window=14, min_periods=14).mean().values  # Smooth chop
    
    # Align to 4h
    vi_plus_norm_aligned = align_htf_to_ltf(prices, df_1d, vi_plus_norm)
    vi_minus_norm_aligned = align_htf_to_ltf(prices, df_1d, vi_minus_norm)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume spike detection on 4h
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vi_plus_norm_aligned[i]) or np.isnan(vi_minus_norm_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20[i]  # Volume spike
        chop_ok = chop_aligned[i] > 50  # Chop > 50 indicates ranging/trading range
        
        if position == 0:
            # Long: VI+ > VI- (uptrend) + chop > 50 (not too choppy) + volume spike
            if (vi_plus_norm_aligned[i] > vi_minus_norm_aligned[i] and 
                chop_ok and 
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Short: VI- > VI+ (downtrend) + chop > 50 + volume spike
            elif (vi_minus_norm_aligned[i] > vi_plus_norm_aligned[i] and 
                  chop_ok and 
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend reversal or volatility drop
            if (vi_minus_norm_aligned[i] > vi_plus_norm_aligned[i] or 
                volume[i] < 0.7 * vol_ma20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend reversal or volatility drop
            if (vi_plus_norm_aligned[i] > vi_minus_norm_aligned[i] or 
                volume[i] < 0.7 * vol_ma20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
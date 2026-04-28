#!/usr/bin/env python3
"""
12h_Vortex_Trend_Plus_Volume
Hypothesis: 12h Vortex indicator (VI+ > VI-) confirms trend direction, combined with volume spike (>2x 20-period MA) for entry.
Vortex catches trends early with less whipsaw than MA crossovers. Works in bull (VI+ > VI-) and bear (VI- > VI+) markets.
Target: 15-25 trades/year to minimize fee drag while capturing sustained trends.
"""

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
    
    # Get daily data for Vortex calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Vortex Indicator (VI) on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Vortex movements
    vm_plus = np.abs(high_1d[1:] - low_1d[:-1])
    vm_minus = np.abs(low_1d[1:] - high_1d[:-1])
    vm_plus = np.concatenate([[np.nan], vm_plus])
    vm_minus = np.concatenate([[np.nan], vm_minus])
    
    # Smooth over 14 periods (standard VI period)
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vm_plus_14 = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus_14 = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    
    # VI+ and VI-
    vi_plus = vm_plus_14 / tr_14
    vi_minus = vm_minus_14 / tr_14
    
    # Align VI to 12h timeframe
    vi_plus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_minus)
    
    # Volume confirmation: >2x 20-period MA on 12h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vi_plus_aligned[i]) or 
            np.isnan(vi_minus_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend direction: VI+ > VI- for uptrend, VI- > VI+ for downtrend
        uptrend = vi_plus_aligned[i] > vi_minus_aligned[i]
        downtrend = vi_minus_aligned[i] > vi_plus_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > (2.0 * vol_ma_20[i])
        
        # Entry logic: trend alignment with volume
        long_entry = vol_confirm and uptrend
        short_entry = vol_confirm and downtrend
        
        # Exit logic: trend reversal
        long_exit = not uptrend  # VI- crosses above VI+
        short_exit = not downtrend  # VI+ crosses above VI-
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Vortex_Trend_Plus_Volume"
timeframe = "12h"
leverage = 1.0
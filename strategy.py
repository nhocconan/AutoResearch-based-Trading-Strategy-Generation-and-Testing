#!/usr/bin/env python3
"""
4d_vortex_volume_regime_v1
Hypothesis: Vortex indicator identifies trend direction, volume confirms strength,
and a chop regime filter avoids whipsaws in ranging markets. Works in bull via
strong upward trends (VI+ > VI-), in bear via strong downward trends (VI- > VI+).
Volume surge filters weak moves, chop filter (>61.8) avoids ranging markets.
Targets ~25-35 trades/year by requiring trend + volume + non-chop confluence.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4d_vortex_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Vortex indicator (14-period)
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    tr = np.maximum(np.abs(high - low), 
                    np.maximum(np.abs(high - np.roll(close, 1)),
                               np.abs(low - np.roll(close, 1))))
    
    # Handle first element
    vm_plus[0] = np.abs(high[0] - low[-1]) if n > 1 else 0
    vm_minus[0] = np.abs(low[0] - high[-1]) if n > 1 else 0
    tr[0] = np.abs(high[0] - low[0])
    
    vi_plus = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values / \
              pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vi_minus = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values / \
               pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Daily data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Chopiness index (14-period) on daily
    atr_daily = np.maximum(np.abs(df_1d['high'].values - df_1d['low'].values),
                           np.maximum(np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1)),
                                      np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))))
    atr_daily[0] = np.abs(df_1d['high'].values[0] - df_1d['low'].values[0])
    sum_atr = pd.Series(atr_daily).rolling(window=14, min_periods=14).sum().values
    highest = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr / (highest - lowest)) / np.log10(14)
    
    # Align chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if data not available
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i]
        
        # Regime filter: avoid choppy markets (chop > 61.8 = ranging)
        non_chop = chop_aligned[i] <= 61.8
        
        if position == 1:  # Long position
            # Exit: trend weakens or chop increases
            if vi_plus[i] <= vi_minus[i] or not non_chop:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend weakens or chop increases
            if vi_minus[i] <= vi_plus[i] or not non_chop:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Strong uptrend + volume + non-chop
            if vi_plus[i] > vi_minus[i] and vol_confirmed and non_chop:
                position = 1
                signals[i] = 0.25
            # Strong downtrend + volume + non-chop
            elif vi_minus[i] > vi_plus[i] and vol_confirmed and non_chop:
                position = -1
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
# 12h_Vortex_Volume_Trend_Signal
# Hypothesis: Uses Vortex Indicator on 1d to determine trend direction (VI+ > VI- = uptrend, VI- > VI+ = downtrend).
# Enters long when VI+ crosses above VI- with volume spike; enters short when VI- crosses above VI+ with volume spike.
# Exits when Vortex trend reverses or volume drops below average. Vortex captures trend initiation, volume confirms strength,
# and 12h timeframe limits trades to avoid fee drag. Works in bull markets by catching strong uptrends and in bear
# markets by catching strong downtrends.

name = "12h_Vortex_Volume_Trend_Signal"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Vortex and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Vortex Indicator (14-period) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first value NaN
    
    # Vortex movements
    vm_plus = np.abs(high_1d[1:] - low_1d[:-1])
    vm_minus = np.abs(low_1d[1:] - high_1d[:-1])
    vm_plus = np.concatenate([[np.nan], vm_plus])
    vm_minus = np.concatenate([[np.nan], vm_minus])
    
    # Sum over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vm_plus_sum = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    
    # VI+ and VI-
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum
    
    # --- 1d volume confirmation (volume > 20-period average) ---
    vol_20_1d = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all 1d indicators to 12h timeframe
    vi_plus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_minus)
    vol_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for Vortex (14)
    start_idx = 14
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(vi_plus_aligned[i]) or
            np.isnan(vi_minus_aligned[i]) or
            np.isnan(vol_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Vortex crossover signals
        vi_plus_cross_above = vi_plus_aligned[i] > vi_minus_aligned[i] and vi_plus_aligned[i-1] <= vi_minus_aligned[i-1]
        vi_minus_cross_above = vi_minus_aligned[i] > vi_plus_aligned[i] and vi_minus_aligned[i-1] <= vi_plus_aligned[i-1]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_20_1d_aligned[i] * 1.5  # 50% above average
        
        if position == 0:
            if vi_plus_cross_above and vol_spike:
                # Long: VI+ crosses above VI- with volume spike
                signals[i] = 0.25
                position = 1
            elif vi_minus_cross_above and vol_spike:
                # Short: VI- crosses above VI+ with volume spike
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: VI- crosses above VI+ (trend reversal)
                if vi_minus_cross_above:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: VI+ crosses above VI- (trend reversal)
                if vi_plus_cross_above:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_vortex_volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Calculate Vortex indicator on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # +VM and -VM
    vm_plus = np.abs(high_1d - np.roll(low_1d, 1))
    vm_minus = np.abs(low_1d - np.roll(high_1d, 1))
    
    # Sum over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vm_plus_sum = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    
    # VI+ and VI-
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum
    
    # Volume confirmation: 4h volume > 1.8x 80-period average
    vol_ma_80 = pd.Series(volume).rolling(window=80, min_periods=80).mean().values
    
    # Align daily Vortex values to 4h timeframe
    vi_plus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_minus)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(vi_plus_aligned[i]) or np.isnan(vi_minus_aligned[i]) or
            np.isnan(vol_ma_80[i])):
            signals[i] = 0.0
            continue
        
        volume_current = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_current > 1.8 * vol_ma_80[i]
        
        # Vortex crossover signals
        vi_plus_prev = vi_plus_aligned[i-1]
        vi_minus_prev = vi_minus_aligned[i-1]
        vi_plus_curr = vi_plus_aligned[i]
        vi_minus_curr = vi_minus_aligned[i]
        
        # Entry conditions
        enter_long = (vi_plus_curr > vi_minus_curr) and (vi_plus_prev <= vi_minus_prev) and vol_confirm
        enter_short = (vi_minus_curr > vi_plus_curr) and (vi_minus_prev <= vi_plus_prev) and vol_confirm
        
        # Exit conditions: opposite crossover
        exit_long = (vi_minus_curr > vi_plus_curr) and (vi_minus_prev <= vi_plus_prev)
        exit_short = (vi_plus_curr > vi_minus_curr) and (vi_plus_prev <= vi_minus_prev)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 4h Vortex crossover strategy with volume confirmation using daily Vortex values.
# Enters long when VI+ crosses above VI- with volume > 1.8x 80-period average.
# Enters short when VI- crosses above VI+ with volume > 1.8x 80-period average.
# Exits on opposite crossover.
# Uses higher volume threshold (1.8x) and longer MA (80) to reduce trade frequency.
# Position size set to 0.25 to manage risk in volatile markets.
# Target: 15-25 trades per year (60-100 total over 4 years) to minimize fee drag.
# Vortex indicator identifies trend initiation and direction, working in both bull and bear markets.
# 4h timeframe provides good balance between signal quality and trade frequency.
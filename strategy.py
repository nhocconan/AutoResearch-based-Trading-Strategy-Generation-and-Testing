#!/usr/bin/env python3
"""
4h_Vortex_Volume_Spike
Hypothesis: 4h Vortex trend + volume spike. Uses directional movement indicator
to capture trends in both bull/bear markets with volume confirmation.
Target: 20-40 trades/year on 4h to avoid fee drag.
"""

name = "4h_Vortex_Volume_Spike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get price, volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Vortex Indicator (VI)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value has no previous close
    
    # VM+ and VM-
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = 0  # First value has no previous low/high
    vm_minus[0] = 0
    
    # Sum over 14 periods
    vi_plus = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vi_minus = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    vt_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # VI+ and VI-
    vi_plus_final = np.divide(vi_plus, vt_sum, out=np.zeros_like(vi_plus), where=vt_sum!=0)
    vi_minus_final = np.divide(vi_minus, vt_sum, out=np.zeros_like(vi_minus), where=vt_sum!=0)
    
    # Volume filter: current volume > 2.0x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need VI calculation (14) and EMA50 (50)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(vi_plus_final[i]) or 
            np.isnan(vi_minus_final[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: VI+ > VI- (bullish trend) AND price above daily EMA50 AND volume spike
            if vi_plus_final[i] > vi_minus_final[i] and close[i] > ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: VI- > VI+ (bearish trend) AND price below daily EMA50 AND volume spike
            elif vi_minus_final[i] > vi_plus_final[i] and close[i] < ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: VI- > VI+ OR price below daily EMA50
            if vi_minus_final[i] > vi_plus_final[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: VI+ > VI- OR price above daily EMA50
            if vi_plus_final[i] > vi_minus_final[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
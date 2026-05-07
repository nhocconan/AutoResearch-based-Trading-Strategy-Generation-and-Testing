#!/usr/bin/env python3
# 4h_Vortex_Trend_Filter_Volume_Spike - Uses Vortex indicator for trend detection with 1d trend filter and volume confirmation
# Designed to work in both bull and bear markets by following strong trends while avoiding whipsaws
# Target: 20-50 trades/year to stay within fee limits

name = "4h_Vortex_Trend_Filter_Volume_Spike"
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
    
    # Load daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Vortex indicator (14-period) on 4h data
    # VM+ = |high - previous low|
    # VM- = |low - previous high|
    # Sum over 14 periods
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    
    # Handle first element
    vm_plus[0] = np.abs(high[0] - low[0])
    vm_minus[0] = vm_plus[0]
    
    # Sum of VM+ and VM- over 14 periods
    sum_vm_plus = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    sum_vm_minus = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    sum_tr = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).rolling(window=14, min_periods=14).sum().values
    sum_tr[0] = high[0] - low[0]  # First TR
    
    # VI+ and VI-
    vi_plus = sum_vm_plus / sum_tr
    vi_minus = sum_vm_minus / sum_tr
    
    # Daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 4-period average
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 14)  # Wait for Vortex and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: VI+ > VI- (bullish trend) with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 2.0
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if vi_plus[i] > vi_minus[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: VI- > VI+ (bearish trend) with volume and daily downtrend
            elif vi_minus[i] > vi_plus[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: trend weakens or volume drops
            if vi_plus[i] <= vi_minus[i] or volume[i] < vol_ma_4[i] * 1.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: trend weakens or volume drops
            if vi_minus[i] <= vi_plus[i] or volume[i] < vol_ma_4[i] * 1.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
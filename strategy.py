#!/usr/bin/env python3
"""
4h_1w_vortex_trend_v1
Hypothesis: 4-hour strategy using weekly Vortex Indicator for trend direction and 4-hour price action for entries.
The Vortex Indicator identifies trend strength and direction: VI+ > VI- indicates uptrend, VI- > VI+ indicates downtrend.
Combined with 4-hour price crossing above/below weekly EMA200 for entry confirmation and volume filter to avoid false signals.
Designed to work in both bull and bear markets by following the weekly trend.
Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Vortex and EMA200
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Vortex Indicator (period=14)
    tr1 = np.maximum(high_1w[1:], low_1w[:-1]) - np.minimum(high_1w[:-1], low_1w[1:])
    tr1 = np.concatenate([[np.nan], tr1])  # align length
    vm1 = np.abs(high_1w[1:] - low_1w[:-1])  # +VM
    vm2 = np.abs(high_1w[:-1] - low_1w[1:])  # -VM
    vm1 = np.concatenate([[np.nan], vm1])
    vm2 = np.concatenate([[np.nan], vm2])
    
    # Sum over 14 periods
    sum_tr1 = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    sum_vm1 = pd.Series(vm1).rolling(window=14, min_periods=14).sum().values
    sum_vm2 = pd.Series(vm2).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    vi_plus = np.where(sum_tr1 != 0, sum_vm1 / sum_tr1, 0)
    vi_minus = np.where(sum_tr1 != 0, sum_vm2 / sum_tr1, 0)
    
    # Weekly EMA200
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly indicators to 4h timeframe
    vi_plus_aligned = align_htf_to_ltf(prices, df_1w, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1w, vi_minus)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # 4-hour volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(vi_plus_aligned[i]) or np.isnan(vi_minus_aligned[i]) or 
            np.isnan(ema200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: VI+ > VI- (uptrend) AND price > EMA200 with volume confirmation
        if (vi_plus_aligned[i] > vi_minus_aligned[i] and close[i] > ema200_1w_aligned[i] and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: VI- > VI+ (downtrend) AND price < EMA200 with volume confirmation
        elif (vi_minus_aligned[i] > vi_plus_aligned[i] and close[i] < ema200_1w_aligned[i] and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: trend reversal or price crosses back below/above EMA200
        elif position == 1 and (vi_minus_aligned[i] >= vi_plus_aligned[i] or close[i] < ema200_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (vi_plus_aligned[i] >= vi_minus_aligned[i] or close[i] > ema200_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1w_vortex_trend_v1"
timeframe = "4h"
leverage = 1.0
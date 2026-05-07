#!/usr/bin/env python3
# 1d_Vortex_Trend_Filter_Vortex
# Hypothesis: Uses Vortex Indicator on weekly timeframe for trend direction,
# combined with 1d price action above/below Vortex and volume confirmation to reduce false signals.
# Vortex identifies trending markets by measuring directional movement.
# Targets 15-25 trades/year to minimize fee drift while maintaining trend-following edge.
# Works in both bull and bear markets by adapting to trend strength via VI+ and VI-.

name = "1d_Vortex_Trend_Filter_Vortex"
timeframe = "1d"
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
    
    # Get weekly data for Vortex calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Vortex Indicator (VI)
    # Parameters: period = 14
    period = 14
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Positive and Negative Vortex Movements
    vm_plus = np.abs(high_1w[1:] - low_1w[:-1])
    vm_minus = np.abs(low_1w[1:] - high_1w[:-1])
    vm_plus = np.concatenate([[np.nan], vm_plus])
    vm_minus = np.concatenate([[np.nan], vm_minus])
    
    # Sum over period
    tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    vm_plus_sum = pd.Series(vm_plus).rolling(window=period, min_periods=period).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=period, min_periods=period).sum().values
    
    # VI+ and VI-
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum
    
    # Align Vortex to 1d timeframe
    vi_plus_1d = align_htf_to_ltf(prices, df_1w, vi_plus)
    vi_minus_1d = align_htf_to_ltf(prices, df_1w, vi_minus)
    
    # Volume confirmation: volume > 20-period average (20 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any critical value is NaN
        if (np.isnan(vi_plus_1d[i]) or np.isnan(vi_minus_1d[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: VI+ > VI- with volume confirmation
            if vi_plus_1d[i] > vi_minus_1d[i] and volume[i] > vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: VI- > VI+ with volume confirmation
            elif vi_minus_1d[i] > vi_plus_1d[i] and volume[i] > vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: VI- crosses above VI+
            if vi_minus_1d[i] > vi_plus_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: VI+ crosses above VI-
            if vi_plus_1d[i] > vi_minus_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
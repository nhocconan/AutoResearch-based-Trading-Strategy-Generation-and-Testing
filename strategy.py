#!/usr/bin/env python3
"""
4h_Vortex_Trend_Filter_v2
Hypothesis: Uses Vortex Indicator on 4h timeframe for trend direction, confirmed by 12h trend filter and volume spike.
Vortex identifies trend strength by comparing positive and negative vortex movements.
Works in both bull and bear markets by capturing established trends with volume confirmation.
Targets 25-35 trades/year to minimize fee drag.
"""

name = "4h_Vortex_Trend_Filter_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Vortex Indicator (14-period)
    # VM+ = |current high - previous low|
    # VM- = |current low - previous high|
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = 0  # first value has no previous
    vm_minus[0] = 0
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR is just high-low
    
    # Sum over 14 periods
    vm_plus_sum = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # VI+ and VI-
    vi_plus = np.divide(vm_plus_sum, tr_sum, out=np.zeros_like(vm_plus_sum), where=tr_sum!=0)
    vi_minus = np.divide(vm_minus_sum, tr_sum, out=np.zeros_like(vm_minus_sum), where=tr_sum!=0)
    
    # 12h trend filter: EMA of 12h close
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if any critical value is NaN
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: VI+ > VI- AND price above 12h EMA with volume spike
            if vi_plus[i] > vi_minus[i] and close[i] > ema_12h_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: VI- > VI+ AND price below 12h EMA with volume spike
            elif vi_minus[i] > vi_plus[i] and close[i] < ema_12h_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: VI- crosses above VI+ (trend weakening) OR price below 12h EMA
            if vi_minus[i] > vi_plus[i] or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: VI+ crosses above VI- (trend weakening) OR price above 12h EMA
            if vi_plus[i] > vi_minus[i] or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
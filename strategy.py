#!/usr/bin/env python3
"""
12h_Vortex_Volume_Spike_Trend_Filter
Hypothesis: 12h Vortex Indicator identifies trend direction, filtered by volume spikes and 1d trend alignment. Works in bull/bear by capturing strong moves with confirmation.
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
    
    # Get 12h data for Vortex calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Vortex Indicator (VI) on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # VM+ and VM-
    vm_plus = np.abs(high_12h[1:] - low_12h[:-1])
    vm_minus = np.abs(low_12h[1:] - high_12h[:-1])
    vm_plus = np.concatenate([[np.nan], vm_plus])
    vm_minus = np.concatenate([[np.nan], vm_minus])
    
    # Sum over 14 periods
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vm_plus_14 = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus_14 = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    
    # VI+ and VI-
    vi_plus = vm_plus_14 / tr14
    vi_minus = vm_minus_14 / tr14
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all higher timeframe data to 12h
    vi_plus_aligned = align_htf_to_ltf(prices, df_12h, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_12h, vi_minus)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Trend filter: price > EMA50 = bullish, < EMA50 = bearish
    d_uptrend = close > ema_50_1d_aligned
    d_downtrend = close < ema_50_1d_aligned
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vi_plus_aligned[i]) or np.isnan(vi_minus_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with trend alignment and volume spike
        # Long: VI+ > VI- + daily uptrend + volume spike
        long_entry = (vi_plus_aligned[i] > vi_minus_aligned[i] and 
                     d_uptrend[i] and 
                     volume_spike[i])
        
        # Short: VI- > VI+ + daily downtrend + volume spike
        short_entry = (vi_minus_aligned[i] > vi_plus_aligned[i] and 
                      d_downtrend[i] and 
                      volume_spike[i])
        
        # Exit on opposite vortex crossover with volume spike
        long_exit = vi_minus_aligned[i] > vi_plus_aligned[i] and volume_spike[i]
        short_exit = vi_plus_aligned[i] > vi_minus_aligned[i] and volume_spike[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Vortex_Volume_Spike_Trend_Filter"
timeframe = "12h"
leverage = 1.0
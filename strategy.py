#!/usr/bin/env python3
"""
4h_4HourDonchian20_SuperTrend_MultiTF_Filter
Hypothesis: 4-hour Donchian(20) breakout combined with 1-day SuperTrend (ATR=10, mult=3)
and volume confirmation provides robust trend-following signals. SuperTrend acts as
a dynamic trend filter to avoid counter-trend trades, while Donchian breakouts capture
momentum. Volume spike confirms institutional participation. Works in both bull and bear
markets by only taking trades in the direction of the higher-timeframe trend.
Target: 20-30 trades/year per symbol.
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
    
    # Get 1d data for SuperTrend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day SuperTrend (ATR=10, multiplier=3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # ATR(10)
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high_1d + low_1d) / 2 + 3 * atr_10
    basic_lb = (high_1d + low_1d) / 2 - 3 * atr_10
    
    # Final Upper and Lower Bands
    final_ub = np.zeros_like(basic_ub)
    final_lb = np.zeros_like(basic_lb)
    for i in range(len(close_1d)):
        if i == 0:
            final_ub[i] = basic_ub[i]
            final_lb[i] = basic_lb[i]
        else:
            if basic_ub[i] < final_ub[i-1] or close_1d[i-1] > final_ub[i-1]:
                final_ub[i] = basic_ub[i]
            else:
                final_ub[i] = final_ub[i-1]
                
            if basic_lb[i] > final_lb[i-1] or close_1d[i-1] < final_lb[i-1]:
                final_lb[i] = basic_lb[i]
            else:
                final_lb[i] = final_lb[i-1]
    
    # SuperTrend
    supertrend = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i == 0:
            supertrend[i] = final_ub[i]
        else:
            if supertrend[i-1] == final_ub[i-1]:
                if close_1d[i] <= final_ub[i]:
                    supertrend[i] = final_ub[i]
                else:
                    supertrend[i] = final_lb[i]
            else:
                if close_1d[i] >= final_lb[i]:
                    supertrend[i] = final_lb[i]
                else:
                    supertrend[i] = final_ub[i]
    
    # SuperTrend trend direction: 1 for uptrend (price > SuperTrend), -1 for downtrend
    supertrend_dir = np.where(close_1d > supertrend, 1, -1)
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_1d, supertrend_dir)
    
    # Get 4h data for Donchian channels (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian(20) upper and lower bands
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe (already in 4h, but ensure proper alignment)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Volume spike detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 20  # need 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_dir_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian High with volume spike and uptrend on 1d SuperTrend
            if (close[i] > donchian_high_aligned[i] and volume_spike[i] and 
                supertrend_dir_aligned[i] == 1):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian Low with volume spike and downtrend on 1d SuperTrend
            elif (close[i] < donchian_low_aligned[i] and volume_spike[i] and 
                  supertrend_dir_aligned[i] == -1):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns below Donchian Low or trend changes to down
            if (close[i] < donchian_low_aligned[i] or 
                supertrend_dir_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns above Donchian High or trend changes to up
            if (close[i] > donchian_high_aligned[i] or 
                supertrend_dir_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_4HourDonchian20_SuperTrend_MultiTF_Filter"
timeframe = "4h"
leverage = 1.0
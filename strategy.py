#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R mean reversion with 1d ADX regime filter and volume confirmation.
In high ADX (>25) trending markets, fade extreme Williams %R readings (<-80 for long, >-20 for short).
In low ADX (<20) ranging markets, use Williams %R for mean reversion at oversold/overbought levels.
Volume spike confirms momentum. Designed for 6h timeframe to capture swings with controlled trade frequency.
Uses discrete position sizing (0.25) to balance return and fee drag.
Target: 50-150 total trades over 4 years = 12-37/year.
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
    
    # Calculate 1d ADX for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_period = 14
    tr_sum = pd.Series(tr).rolling(window=tr_period, min_periods=tr_period).sum().values
    dm_plus_sum = pd.Series(dm_plus).rolling(window=tr_period, min_periods=tr_period).sum().values
    dm_minus_sum = pd.Series(dm_minus).rolling(window=tr_period, min_periods=tr_period).sum().values
    
    # Avoid division by zero
    tr_sum[tr_sum == 0] = 1e-10
    
    di_plus = 100 * dm_plus_sum / tr_sum
    di_minus = 100 * dm_minus_sum / tr_sum
    
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=tr_period, min_periods=tr_period).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h Williams %R
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Avoid division by zero
    hh_ll = highest_high - lowest_low
    hh_ll[hh_ll == 0] = 1e-10
    
    williams_r = -100 * (highest_high - close) / hh_ll
    
    # Calculate volume spike: current volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(period, 20, tr_period*2)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        wr = williams_r[i]
        vol_spike = volume_spike[i]
        
        # Regime determination
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        if position == 0:
            # Long conditions
            long_condition = False
            if is_trending:
                # In trend: fade extreme oversold
                long_condition = wr < -80 and vol_spike
            elif is_ranging:
                # In range: mean reversion from oversold
                long_condition = wr < -70 and vol_spike
            
            # Short conditions
            short_condition = False
            if is_trending:
                # In trend: fade extreme overbought
                short_condition = wr > -20 and vol_spike
            elif is_ranging:
                # In range: mean reversion from overbought
                short_condition = wr > -30 and vol_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            if position == 1:
                # Exit long: Williams %R returns to overbought or adverse extreme
                exit_signal = wr > -20 or (is_trending and wr < -85)
            elif position == -1:
                # Exit short: Williams %R returns to oversold or adverse extreme
                exit_signal = wr < -80 or (is_trending and wr > -15)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_MeanReversion_1dADXRegime_VolumeSpike"
timeframe = "6h"
leverage = 1.0
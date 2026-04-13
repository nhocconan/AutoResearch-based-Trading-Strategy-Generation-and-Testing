#!/usr/bin/env python3
"""
4h_1d_donchian_breakout_volume
Hypothesis: On 4h timeframe, price breaking above/below Donchian(20) with volume expansion (>1.5x 20-bar avg) and daily ATR filter (>0.015) captures momentum moves. Works in bull (breakouts continue) and bear (breakouts rarer but stronger when they occur). Target: 20-50 trades/year to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR (14-period) on daily for volatility filter
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    volatility_filter = atr > 0.015  # Only trade when volatility is sufficient
    
    # Volume expansion: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    # Align all signals to 4h timeframe
    high_max_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high}), high_max)
    low_min_aligned = align_htf_to_ltf(prices, pd.DataFrame({'low': low}), low_min)
    volatility_filter_aligned = align_htf_to_ltf(prices, df_1d, volatility_filter.astype(float))
    volume_expansion_aligned = align_htf_to_ltf(prices, pd.DataFrame({'volume': volume}), volume_expansion.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(high_max_aligned[i]) or 
            np.isnan(low_min_aligned[i]) or 
            np.isnan(volatility_filter_aligned[i]) or 
            np.isnan(volume_expansion_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Donchian breakout with volume and volatility
        long_break = close[i] > high_max_aligned[i]
        short_break = close[i] < low_min_aligned[i]
        
        long_entry = long_break and volume_expansion_aligned[i] > 0.5 and volatility_filter_aligned[i] > 0.5
        short_entry = short_break and volume_expansion_aligned[i] > 0.5 and volatility_filter_aligned[i] > 0.5
        
        # Exit when price returns to opposite Donchian level (mean reversion)
        exit_long = position == 1 and close[i] <= low_min_aligned[i]
        exit_short = position == -1 and close[i] >= high_max_aligned[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_breakout_volume"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
"""
4h Donchian Breakout with 1d Volume and ADX Filter
Hypothesis: Donchian channel breakouts capture strong trends. Volume confirmation and 1d ADX filter ensure trades occur only in high-momentum, trending regimes, reducing whipsaws in chop. Works in bull/bear by trading breakouts in direction of higher-timeframe trend. Target: ~25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d ADX for trend strength filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range (1d)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr_1d[0]
    
    # Directional Movement (1d)
    dm_plus_1d = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                          np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus_1d = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                           np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus_1d[0] = 0
    dm_minus_1d[0] = 0
    
    # Smoothed values (14-period)
    tr14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    dm_plus14_1d = pd.Series(dm_plus_1d).rolling(window=14, min_periods=14).sum().values
    dm_minus14_1d = pd.Series(dm_minus_1d).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus_1d = np.where(tr14_1d > 0, 100 * dm_plus14_1d / tr14_1d, 0)
    di_minus_1d = np.where(tr14_1d > 0, 100 * dm_minus14_1d / tr14_1d, 0)
    
    # DX and ADX
    dx_1d = np.where((di_plus_1d + di_minus_1d) > 0, 100 * np.abs(di_plus_1d - di_minus_1d) / (di_plus_1d + di_minus_1d), 0)
    adx_1d = pd.Series(dx_1d).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for Donchian and ADX
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_val = adx_1d_aligned[i]
        vol_confirmed = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long breakout: price above upper Donchian + volume + strong 1d trend
            if price > highest_high[i] and vol_confirmed and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price below lower Donchian + volume + strong 1d trend
            elif price < lowest_low[i] and vol_confirmed and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long if price breaks below lower Donchian or trend weakens
            if price < lowest_low[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price breaks above upper Donchian or trend weakens
            if price > highest_high[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_1dADX_Volume"
timeframe = "4h"
leverage = 1.0
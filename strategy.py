#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + Volume Spike + ADX Filter
Breakout of 20-period high/low with volume confirmation and ADX > 25 for trend strength.
Designed for 4h timeframe to capture trending moves with low trade frequency.
Works in both bull and bear markets by filtering for strong trends.
"""

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
    
    # Calculate Donchian channels (20-period high/low)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ADX for trend filter (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high - np.concatenate([[high[0]], high[:-1]])) > (np.concatenate([[low[0]], low[:-1]]) - low), 
                       np.maximum(high - np.concatenate([[high[0]], high[:-1]]), 0), 0)
    dm_minus = np.where((np.concatenate([[low[0]], low[:-1]]) - low) > (high - np.concatenate([[high[0]], high[:-1]])), 
                        np.maximum(np.concatenate([[low[0]], low[:-1]]) - low, 0), 0)
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 40  # need enough history for ADX calculation
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: break above Donchian high with volume spike and ADX > 25
            if (price > donchian_high[i] and volume_spike[i] and adx[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume spike and ADX > 25
            elif (price < donchian_low[i] and volume_spike[i] and adx[i] > 25):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price breaks below Donchian low or ADX < 20 (trend weakening)
            if price < donchian_low[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price breaks above Donchian high or ADX < 20 (trend weakening)
            if price > donchian_high[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_VolumeSpike_ADXFilter"
timeframe = "4h"
leverage = 1.0
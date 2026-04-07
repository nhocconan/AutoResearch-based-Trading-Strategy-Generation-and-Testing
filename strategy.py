#!/usr/bin/env python3
"""
4h Donchian Breakout with 1d Volume Confirmation and ADX Trend Filter
Long when price breaks above Donchian(20) high + volume spike + ADX>25
Short when price breaks below Donchian(20) low + volume spike + ADX>25
Exit when price crosses opposite Donchian boundary or ADX<20 (range)
Target: 20-40 trades/year per symbol with strong trend capture
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_volume_adx_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Donchian Channels (20-period) ===
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === ADX Calculation (14-period) ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # === 1d Volume Spike Detector ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (vol_ma_1d * 1.5)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any NaN values
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian OR ADX weakens (<20)
            if close[i] < low_20[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian OR ADX weakens (<20)
            if close[i] > high_20[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: break above upper Donchian + volume spike + strong trend (ADX>25)
            if (close[i] > high_20[i] and 
                vol_spike_1d_aligned[i] and 
                adx[i] > 25):
                position = 1
                signals[i] = 0.25
            # Short: break below lower Donchian + volume spike + strong trend (ADX>25)
            elif (close[i] < low_20[i] and 
                  vol_spike_1d_aligned[i] and 
                  adx[i] > 25):
                position = -1
                signals[i] = -0.25
    
    return signals
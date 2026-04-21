#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeATRFilter_Tight
Hypothesis: Donchian(20) breakouts with volume confirmation and ATR-based stoploss work on 4h timeframe for BTC and ETH in both bull and bear markets. The strategy uses 1d timeframe for volume confirmation and ATR calculation to reduce noise, targeting 20-50 trades/year per symbol (80-200 over 4 years). Tight entry conditions prevent overtrading and fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for volume and ATR calculation (less noisy than 4h)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on 4h close
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian upper/lower (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d ATR for stoploss (more stable)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1d volume average for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current 4h volume > 1.5x 1d average volume
        volume_ok = volume > 1.5 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume
            if price > donchian_high[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume
            elif price < donchian_low[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price reaches Donchian low or stoploss
            if price <= donchian_low[i] or price < prices['close'].iloc[i-1] - 2.0 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price reaches Donchian high or stoploss
            if price >= donchian_high[i] or price > prices['close'].iloc[i-1] + 2.0 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_VolumeATRFilter_Tight"
timeframe = "4h"
leverage = 1.0
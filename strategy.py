#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d ATR-based volatility breakout + volume confirmation + Donchian(20) trend filter.
Long when price breaks above Donchian(20) high with ATR(14) > 1.2x its 50-period MA and volume > 1.5x 20-period MA.
Short when price breaks below Donchian(20) low with same volatility/volume filters.
Volatility breakout captures expansion phases; volume confirmation reduces false signals; Donchian filter ensures trend alignment.
Target: 80-160 total trades over 4 years (20-40/year) to minimize fee drag. Uses discrete sizing 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ATR(14)
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first TR is NaN
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR(14) 50-period MA for volatility filter
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    
    # Calculate 1d volume 20-period MA
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Get 4h data for Donchian(20)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian(20) channels
    donch_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align all indicators to primary 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    donch_high_20_aligned = align_htf_to_ltf(prices, df_4h, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_4h, donch_low_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # need enough for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_14_aligned[i]) or np.isnan(atr_ma_50_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(volume_1d_aligned[i]) or
            np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current ATR > 1.2x its 50-period MA
        volatility_expansion = atr_14_aligned[i] > 1.2 * atr_ma_50_aligned[i]
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA
        volume_confirmed = volume_1d_aligned[i] > 1.5 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian(20) high with volatility expansion and volume
            if (close[i] > donch_high_20_aligned[i] and 
                volatility_expansion and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian(20) low with volatility expansion and volume
            elif (close[i] < donch_low_20_aligned[i] and 
                  volatility_expansion and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below Donchian(20) low or volatility contraction
            if (close[i] < donch_low_20_aligned[i] or 
                not volatility_expansion):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above Donchian(20) high or volatility contraction
            if (close[i] > donch_high_20_aligned[i] or 
                not volatility_expansion):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dATR_VolumeBreakout_Donchian20"
timeframe = "4h"
leverage = 1.0
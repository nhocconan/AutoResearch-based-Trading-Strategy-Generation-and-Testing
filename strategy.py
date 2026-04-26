#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_ATRRegime_VolumeFilter
Hypothesis: Donchian(20) breakouts filtered by ATR-based regime (low volatility) and volume confirmation (>1.5x 20-bar MA). Designed for 4h timeframe to capture breakouts during consolidation periods with institutional participation. Works in bull/bear markets by using ATR regime to avoid whipsaws in ranging markets and volume to confirm breakout validity. Target: 15-25 trades/year (60-100 total over 4 years).
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
    
    # Load 1d data ONCE before loop for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ATR(14) for volatility regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 1d ATR ratio: current ATR / 20-period ATR mean (regime filter)
    atr_ma_20 = pd.Series(atr_14_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_14_1d / atr_ma_20
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Donchian(20) channels on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size (25% of capital)
    
    # Warmup: max of calculations (20 for Donchian/vol, 34 for ATR regime)
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        upper_channel = donchian_high[i]
        lower_channel = donchian_low[i]
        mid_channel = donchian_mid[i]
        atr_ratio_val = atr_ratio_aligned[i]
        vol_spike = volume_spike[i]
        
        # Regime filter: low volatility environment (ATR ratio < 0.8) = consolidation
        low_volatility = atr_ratio_val < 0.8
        
        # Entry conditions: Donchian breakout in low volatility regime with volume confirmation
        long_entry = (close_val > upper_channel) and low_volatility and vol_spike
        short_entry = (close_val < lower_channel) and low_volatility and vol_spike
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
            elif short_entry:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on mean reversion to mid-channel or volatility expansion
            if close_val < mid_channel or atr_ratio_val > 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = base_size
        elif position == -1:
            # Short - exit on mean reversion to mid-channel or volatility expansion
            if close_val > mid_channel or atr_ratio_val > 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Donchian20_Breakout_ATRRegime_VolumeFilter"
timeframe = "4h"
leverage = 1.0
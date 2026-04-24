#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R extreme reversal with 1d ATR regime filter and volume confirmation.
- Long when Williams %R(14) < -80 (oversold) and ATR(14) < ATR(50) (low volatility regime) and volume > 1.5x average
- Short when Williams %R(14) > -20 (overbought) and ATR(14) < ATR(50) (low volatility regime) and volume > 1.5x average
- Exit when Williams %R returns to -50 (mean reversion) or ATR regime changes (volatility expansion)
- Uses 1d HTF for ATR regime (proven effective across multiple experiments)
- Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
- Designed to work in both bull and bear markets via mean reversion in ranging conditions and volatility filter
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
    
    # Calculate Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and ATR(50) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], 
                           np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_50_1d = pd.Series(tr_1d).rolling(window=50, min_periods=50).mean().values
    
    # ATR regime: low volatility when ATR(14) < ATR(50)
    atr_regime_low_vol = atr_14_1d < atr_50_1d
    
    # Align 1d indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime_low_vol.astype(float))
    
    # Volume confirmation: > 1.5x 30-period average volume
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 50, 30) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(atr_regime_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80), low volatility regime, volume spike
            if williams_r_aligned[i] < -80 and atr_regime_aligned[i] > 0.5 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20), low volatility regime, volume spike
            elif williams_r_aligned[i] > -20 and atr_regime_aligned[i] > 0.5 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R returns to -50 (mean reversion) or volatility expands
            if williams_r_aligned[i] >= -50 or atr_regime_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns to -50 (mean reversion) or volatility expands
            if williams_r_aligned[i] <= -50 or atr_regime_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dATRRegime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0
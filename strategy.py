#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Donchian breakout captures momentum in trending markets (works in both bull/bear)
- 1d ATR regime: only trade when ATR(14) > 20-period median ATR (high volatility regimes)
- Volume confirmation: volume > 1.8x 20-period average to avoid false breakouts
- Exit: Donchian(10) opposite breakout or ATR regime flip to low volatility
- Uses discrete sizing ±0.25 to minimize fee churn
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
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
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ATR(14) for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    # ATR(14)
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 20-period median ATR for regime classification
    atr_median_20 = pd.Series(atr_14_1d).rolling(window=20, min_periods=20).median().values
    
    # Align ATR and median ATR to 4h timeframe
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_median_20_aligned = align_htf_to_ltf(prices, df_1d, atr_median_20)
    
    # Calculate 4h Donchian channels (20-period for entry, 10-period for exit)
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 20)  # Need 20 for Donchian, 14 for ATR, 20 for median
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(atr_median_20_aligned[i]) or
            np.isnan(donchian_high_20[i]) or
            np.isnan(donchian_low_20[i]) or
            np.isnan(donchian_high_10[i]) or
            np.isnan(donchian_low_10[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility regime filter: only trade in high volatility regimes
        high_vol_regime = atr_14_1d_aligned[i] > atr_median_20_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above 20-period Donchian high + volume + high vol regime
            if (close[i] > donchian_high_20[i] and 
                volume_confirm and 
                high_vol_regime):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period Donchian low + volume + high vol regime
            elif (close[i] < donchian_low_20[i] and 
                  volume_confirm and 
                  high_vol_regime):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below 10-period Donchian low OR low volatility regime
            if (close[i] < donchian_low_10[i] or 
                not high_vol_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above 10-period Donchian high OR low volatility regime
            if (close[i] > donchian_high_10[i] or 
                not high_vol_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dATRRegime_VolumeConfirm"
timeframe = "4h"
leverage = 1.0
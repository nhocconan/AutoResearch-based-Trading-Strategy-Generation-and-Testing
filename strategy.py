#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Donchian(20) breakouts capture momentum in both bull and bear markets.
- 1d ATR regime: only trade when ATR(14) > 30-period median ATR (high volatility regimes).
- Volume confirmation: >1.5x 20-period average volume reduces false breakouts.
- Discrete position sizing (0.25) minimizes fee drag.
- Target trades: 75-200 total over 4 years (19-50/year) on 4h timeframe.
- Works in bull via breakouts, bear via volatility filters avoiding whipsaws.
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
    
    # Get 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ATR(14) and its 30-period median for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0  # first bar has no previous close
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_median_30 = pd.Series(atr_14).rolling(window=30, min_periods=30).median().values
    
    # Align ATR regime to 4h timeframe (using previous completed 1d bar)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_median_30_aligned = align_htf_to_ltf(prices, df_1d, atr_median_30)
    high_volatility = atr_14_aligned > atr_median_30_aligned
    
    # Donchian(20) channels on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 30) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(atr_median_30_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian high with volume confirm and high volatility regime
            if close[i] > donchian_high[i] and volume_confirm[i] and high_volatility[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume confirm and high volatility regime
            elif close[i] < donchian_low[i] and volume_confirm[i] and high_volatility[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below Donchian low OR low volatility regime
            if close[i] < donchian_low[i] or not high_volatility[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian high OR low volatility regime
            if close[i] > donchian_high[i] or not high_volatility[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_ATRRegime_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume spike confirmation.
- Long when price breaks above Donchian upper (20-period high) AND 1d ATR ratio < 0.8 (low volatility regime) AND volume > 1.5 * 20-bar median volume
- Short when price breaks below Donchian lower (20-period low) AND 1d ATR ratio < 0.8 AND volume > 1.5 * 20-bar median volume
- Exit on opposite Donchian breakout or when 1d ATR ratio > 1.2 (high volatility regime) to avoid whipsaws in choppy markets
- Uses 4h primary timeframe with 1d HTF to target 75-200 total trades over 4 years (19-50/year)
- Donchian channels provide clear trend-following structure that works in both bull and bear markets
- 1d ATR ratio (ATR(7)/ATR(30)) filters for low-volatility environments where breakouts are more reliable
- Volume spike confirmation reduces false breakouts from low-volume noise
- Designed for BTC/ETH with edge in capturing sustained moves during low-volatility regimes
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
    
    # Calculate Donchian channels (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(7) and ATR(30) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_7 = pd.Series(tr).rolling(window=7, min_periods=7).mean().values
    atr_30 = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    
    # Avoid division by zero
    atr_ratio = np.where(atr_30 > 0, atr_7 / atr_30, 1.0)
    
    # Align 1d ATR ratio to 4h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Volume spike filter: volume > 1.5 * median volume of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (1.5 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 30) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper, low volatility regime, volume spike
            if close[i] > donchian_upper[i] and atr_ratio_aligned[i] < 0.8 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower, low volatility regime, volume spike
            elif close[i] < donchian_lower[i] and atr_ratio_aligned[i] < 0.8 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian lower OR high volatility regime (ATR ratio > 1.2)
            if close[i] < donchian_lower[i] or atr_ratio_aligned[i] > 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian upper OR high volatility regime (ATR ratio > 1.2)
            if close[i] > donchian_upper[i] or atr_ratio_aligned[i] > 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATRRegime_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0
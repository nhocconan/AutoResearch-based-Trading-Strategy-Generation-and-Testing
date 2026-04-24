#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Long when price breaks above Donchian upper (20-period high) AND 1d ATR(14) > 1d ATR(50) (high volatility regime)
- Short when price breaks below Donchian lower (20-period low) AND 1d ATR(14) > 1d ATR(50) (high volatility regime)
- Volume confirmation: current volume > 1.5 * 20-period average volume
- Exit on opposite Donchian level (exit long on lower, exit short on upper)
- Uses 4h primary with 1d HTF to target 75-200 total trades over 4 years (19-50/year)
- Donchian provides clear breakout levels; ATR regime ensures trading only in high volatility; volume confirms momentum
- Designed to work in both bull (breakouts with momentum) and bear (breakouts with momentum) markets
- Signal size: 0.25 discrete levels to minimize fee churn
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
    
    # Calculate Donchian channels (20-period) on 4h data
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Load 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and ATR(50) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) and ATR(50)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Regime filter: high volatility when ATR(14) > ATR(50)
    high_vol_regime = atr_14 > atr_50
    
    # Align 1d indicators to 4h timeframe
    high_vol_regime_aligned = align_htf_to_ltf(prices, df_1d, high_vol_regime)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 20) + 1  # Need Donchian(20), ATR(50), and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(high_vol_regime_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian upper AND high volatility regime AND volume confirmation
            if close[i] > donchian_upper[i] and high_vol_regime_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian lower AND high volatility regime AND volume confirmation
            elif close[i] < donchian_lower[i] and high_vol_regime_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below Donchian lower (opposite level)
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above Donchian upper (opposite level)
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATRRegime_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0
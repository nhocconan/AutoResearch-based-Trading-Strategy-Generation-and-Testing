#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Long when price breaks above Donchian upper (20-period high) AND 1d ATR(14) > 1.5 * ATR(50) (high volatility regime)
- Short when price breaks below Donchian lower (20-period low) AND 1d ATR(14) > 1.5 * ATR(50) (high volatility regime)
- Volume confirmation: current volume > 1.8 * 20-period average volume
- Exit on opposite Donchian level (exit long on lower, exit short on upper)
- Uses 4h primary with 1d HTF to target 75-200 total trades over 4 years (19-50/year)
- Donchian provides clear breakout levels; ATR regime filters for high momentum environments; volume confirms conviction
- Works in both bull (breakouts with momentum) and bear (breakdowns with momentum) markets
- Signal size: 0.25 discrete levels to minimize fee churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and ATR(50) for regime filter
    # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_close = df_1d['close'].shift(1)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - prev_close)
    tr3 = abs(df_1d['low'] - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    atr_50 = tr.rolling(window=50, min_periods=50).mean().values
    
    # High volatility regime: short-term ATR > 1.5 * long-term ATR
    high_vol_regime = atr_14 > (1.5 * atr_50)
    
    # Align 1d ATR regime to 4h timeframe
    high_vol_aligned = align_htf_to_ltf(prices, df_1d, high_vol_regime)
    
    # Calculate Donchian channels (20-period) on 4h data
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20) + 1  # Need ATR50 and Donchian20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(high_vol_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian upper AND high vol regime AND volume confirmation
            if close[i] > donchian_upper[i] and high_vol_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian lower AND high vol regime AND volume confirmation
            elif close[i] < donchian_lower[i] and high_vol_aligned[i] and volume_confirm[i]:
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
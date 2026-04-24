#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Long when price breaks above Donchian upper (20-period high) AND 1d ATR(14) > 1d ATR(50) (high volatility regime) AND volume > 1.5x 20-period average
- Short when price breaks below Donchian lower (20-period low) AND 1d ATR(14) > 1d ATR(50) AND volume > 1.5x 20-period average
- Uses 12h primary timeframe with 1d HTF to target 50-150 trades over 4 years (12-37/year)
- Donchian breakouts capture momentum in both bull and bear markets
- ATR regime filter ensures trades only occur in sufficient volatility environments (avoids low-vol whipsaws)
- Volume confirmation reduces false breakouts
- Designed for BTC/ETH with edge in trending markets (works in both bull continuation and bear acceleration)
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
    
    # Calculate Donchian channels using previous 20 periods (no look-ahead)
    # Upper band: highest high of previous 20 periods
    # Lower band: lowest low of previous 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Get 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
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
    
    # ATR regime: ATR(14) > ATR(50) indicates increasing volatility
    atr_regime = atr_14 > atr_50
    
    # Align 1d indicators to 12h timeframe
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime)
    
    # Volume confirmation: > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(atr_regime_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper, high volatility regime, volume confirmation
            if close[i] > donchian_upper[i] and atr_regime_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower, high volatility regime, volume confirmation
            elif close[i] < donchian_lower[i] and atr_regime_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below Donchian lower (mean reversion) OR volatility drops
            if close[i] < donchian_lower[i] or not atr_regime_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian upper (mean reversion) OR volatility drops
            if close[i] > donchian_upper[i] or not atr_regime_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dATRRegime_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0
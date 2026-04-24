#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R mean reversion with 1d ADX regime filter and volume confirmation.
- Long when Williams %R(14) < -80 (oversold) AND ADX(14) < 25 (ranging market) AND volume > 1.5x SMA20 volume
- Short when Williams %R(14) > -20 (overbought) AND ADX(14) < 25 (ranging market) AND volume > 1.5x SMA20 volume
- Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts)
- Uses 12h primary timeframe with 1d HTF for ADX regime to target 50-150 trades over 4 years (12-37/year)
- Williams %R identifies extreme price levels for mean reversion in ranging markets
- ADX filter ensures we only mean revert in low-trend environments, avoiding whipsaws in strong trends
- Volume confirmation reduces false signals by requiring participation
- Designed for BTC/ETH with edge in ranging markets (2022-2024, 2025) where mean reversion works
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
    
    # Calculate Williams %R(14) using previous period (no look-ahead)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().shift(1).values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Get 1d data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 1.5x SMA20 volume
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > 1.5 * volume_sma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_sma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80), ranging market (ADX < 25), volume confirmation
            if williams_r[i] < -80 and adx_aligned[i] < 25 and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20), ranging market (ADX < 25), volume confirmation
            elif williams_r[i] > -20 and adx_aligned[i] < 25 and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -50 (mean reversion complete)
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -50 (mean reversion complete)
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_MeanReversion_1dADXRegime_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0
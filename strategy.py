#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian channel breakout with 1d ATR regime filter and volume confirmation.
- Long when price breaks above Donchian(20) high AND ATR(14) < ATR(50) (low volatility regime) AND volume > 1.5 * median volume
- Short when price breaks below Donchian(20) low AND ATR(14) < ATR(50) (low volatility regime) AND volume > 1.5 * median volume
- Exit on opposite Donchian breakout or volatility expansion (ATR(14) > ATR(50))
- Uses 12h primary timeframe with 1d HTF to target 50-150 total trades over 4 years (12-37/year)
- Donchian breakouts capture momentum in both bull and bear markets
- ATR regime filter avoids whipsaws during high volatility periods (works in 2022 crash and 2025 bear)
- Volume confirmation ensures breakout validity
- Designed for BTC/ETH with edge in breakout continuation during low volatility regimes
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
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    donchian_high = rolling_max(high, 20)
    donchian_low = rolling_min(low, 20)
    
    # Get 1d data ONCE before loop for ATR regime filter
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
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Align 1d ATR values to 12h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    
    # Volume confirmation: volume > 1.5 * median volume of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (1.5 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i]) or 
            np.isnan(vol_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high, low volatility regime, volume confirmation
            if close[i] > donchian_high[i] and atr_14_aligned[i] < atr_50_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, low volatility regime, volume confirmation
            elif close[i] < donchian_low[i] and atr_14_aligned[i] < atr_50_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low OR volatility expansion (ATR14 > ATR50)
            if close[i] < donchian_low[i] or atr_14_aligned[i] > atr_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high OR volatility expansion (ATR14 > ATR50)
            if close[i] > donchian_high[i] or atr_14_aligned[i] > atr_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dATRRegime_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0
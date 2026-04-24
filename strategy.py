#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Long when price breaks above Donchian(20) high AND 1d ATR(14) < 1d ATR(50) (low volatility regime) AND volume > 1.5 * median volume
- Short when price breaks below Donchian(20) low AND 1d ATR(14) < 1d ATR(50) (low volatility regime) AND volume > 1.5 * median volume
- Exit when price crosses Donchian(20) midpoint OR volatility regime shifts to high (ATR14 > ATR50)
- Uses 4h primary timeframe with 1d HTF to target 75-200 total trades over 4 years (19-50/year)
- Donchian breakouts capture momentum bursts in low volatility regimes
- ATR regime filter avoids whipsaws during high volatility periods
- Volume confirmation ensures breakout validity
- Designed for BTC/ETH with edge in both trending and ranging markets
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
    
    # Calculate Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Get 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and ATR(50)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Align 1d ATR values to 4h timeframe
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
        
        # Low volatility regime: ATR14 < ATR50
        low_vol_regime = atr_14_aligned[i] < atr_50_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high, low volatility regime, volume confirmation
            if close[i] > donchian_high[i] and low_vol_regime and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, low volatility regime, volume confirmation
            elif close[i] < donchian_low[i] and low_vol_regime and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses Donchian mid OR volatility regime shifts to high
            if close[i] < donchian_mid[i] or not low_vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses Donchian mid OR volatility regime shifts to high
            if close[i] > donchian_mid[i] or not low_vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATRRegime_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0
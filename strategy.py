#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR-based trend filter and volume confirmation.
- Long when price breaks above Donchian upper AND 1d ATR(14) > median ATR(50) (high volatility regime)
- Short when price breaks below Donchian lower AND 1d ATR(14) > median ATR(50) (high volatility regime)
- Volume must be > 1.5 * median volume of last 20 bars (volume confirmation)
- Exit on opposite Donchian breakout or when ATR(14) < 0.8 * median ATR(50) (low volatility exit)
- Uses 12h primary timeframe with 1d HTF to target 50-150 total trades over 4 years (12-37/year)
- Donchian channels provide clear breakout levels that work in both trending and ranging markets
- ATR filter ensures we only trade during high volatility periods, reducing false breakouts
- Volume confirmation adds robustness to breakout signals
- Designed for BTC/ETH with edge in capturing strong moves during volatile periods while avoiding chop
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
    
    # Calculate Donchian channels (20-period) based on previous bar
    # Upper = max(high of last 20 bars), Lower = min(low of last 20 bars)
    lookback = 20
    upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    
    # Get 1d data ONCE before loop for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and its 50-period median for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan  # First bar has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_median_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).median().values
    
    # Align 1d ATR and its median to 12h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_median_50_aligned = align_htf_to_ltf(prices, df_1d, atr_median_50)
    
    # Volume confirmation: volume > 1.5 * median volume of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (1.5 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_median_50_aligned[i]) or 
            np.isnan(vol_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper, high volatility regime, volume confirmation
            if close[i] > upper[i] and atr_14_aligned[i] > atr_median_50_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower, high volatility regime, volume confirmation
            elif close[i] < lower[i] and atr_14_aligned[i] > atr_median_50_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian lower OR low volatility regime
            if close[i] < lower[i] or atr_14_aligned[i] < 0.8 * atr_median_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian upper OR low volatility regime
            if close[i] > upper[i] or atr_14_aligned[i] < 0.8 * atr_median_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dATR_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0
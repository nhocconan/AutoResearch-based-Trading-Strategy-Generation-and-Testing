#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Long: price breaks above Donchian upper (20) + ATR(14)/ATR(50) < 0.8 (low vol regime) + volume > 1.5x 20-period avg
- Short: price breaks below Donchian lower (20) + ATR(14)/ATR(50) < 0.8 (low vol regime) + volume > 1.5x 20-period avg
- Exit: price re-enters Donchian channel (mean reversion) OR ATR regime shifts to high volatility (ATR ratio > 1.2)
- Uses Donchian for structure, ATR regime to avoid whipsaws in high volatility, volume for conviction
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull (buy breakouts in low vol uptrend) and bear (sell breakdowns in low vol downtrend)
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
    
    # Volume confirmation: > 1.5x 20-period average (balanced to avoid overtrading)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ATR for regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr2[0] = high_1d[0] - close_1d[0]  # First period
    tr3[0] = low_1d[0] - close_1d[0]   # First period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR calculations
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_50  # Ratio < 0.8 = low volatility regime
    
    # Align ATR ratio to 4h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for ATR50, 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Low volatility regime filter (ATR ratio < 0.8)
        low_vol_regime = atr_ratio_aligned[i] < 0.8
        
        if position == 0:
            # Long: price breaks above Donchian high + volume confirmation + low vol regime
            if (close[i] > donchian_high[i] and 
                volume_confirm and 
                low_vol_regime):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume confirmation + low vol regime
            elif (close[i] < donchian_low[i] and 
                  volume_confirm and 
                  low_vol_regime):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price re-enters below Donchian high (mean reversion) OR high volatility regime (ATR ratio > 1.2)
            if close[i] < donchian_high[i] or atr_ratio_aligned[i] > 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters above Donchian low (mean reversion) OR high volatility regime (ATR ratio > 1.2)
            if close[i] > donchian_low[i] or atr_ratio_aligned[i] > 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dATRRegime_VolumeConfirm"
timeframe = "4h"
leverage = 1.0
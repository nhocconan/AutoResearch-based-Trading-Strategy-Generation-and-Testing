#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume spike confirmation.
- Donchian(20) breakout captures strong momentum moves in both bull and bear markets
- 1d ATR regime filter: ATR(14) > 1.2 * ATR(50) indicates high volatility/trending regime (favor breakouts)
- Volume confirmation: > 2.0x 20-period average ensures breakout validity
- Discrete position size 0.25 to minimize drawdown during crashes like 2022
- Target: 20-40 trades/year on 4h timeframe (80-160 total over 4 years)
- Uses tighter volume confirmation (2.0x) and volatility regime filter to reduce overtrading
- Optimized for BTC/ETH performance by avoiding low-volatility choppy markets where breakouts fail
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
    
    # Donchian(20) channels (using prior bar to avoid look-ahead)
    high_shifted = np.roll(high, 1)
    low_shifted = np.roll(low, 1)
    high_shifted[0] = np.nan
    low_shifted[0] = np.nan
    
    # 20-period highest high and lowest low from prior bar
    highest_high = pd.Series(high_shifted).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_shifted).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) and ATR(50) for regime filter
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Regime: ATR(14) > 1.2 * ATR(50) indicates high volatility/trending market
    atr_regime = atr_14 > (1.2 * atr_50)
    
    # Align 1d ATR regime to 4h
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Donchian, ATR(50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_regime_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average) and volatility regime
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        in_trending_regime = atr_regime_aligned[i] > 0.5  # boolean as float
        
        if position == 0:
            # Long: Breakout above Donchian upper band AND volume confirmation AND trending regime
            if close[i] > highest_high[i] and volume_confirm and in_trending_regime:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below Donchian lower band AND volume confirmation AND trending regime
            elif close[i] < lowest_low[i] and volume_confirm and in_trending_regime:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close below Donchian lower band (mean reversion) OR loss of momentum
            if close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close above Donchian upper band (mean reversion) OR loss of momentum
            if close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dATRRegime_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0
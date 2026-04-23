#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Primary timeframe: 6h, HTF: 1d for ATR regime (trend strength)
- Long: Close breaks above Donchian upper (20-bar high) + ATR(1d) > ATR MA(50) (high volatility regime) + volume > 1.5x 20-period avg
- Short: Close breaks below Donchian lower (20-bar low) + ATR(1d) > ATR MA(50) + volume > 1.5x 20-period avg
- Exit: Close reverts to Donchian midpoint (mean of 20-bar high/low)
- Uses ATR regime to avoid low-volatility choppy markets where breakouts fail
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in both bull and bear markets: breakouts work in trending markets, ATR filter ensures sufficient volatility
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
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period) on 6h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_20 + lowest_20) / 2.0
    
    # 1d ATR for regime filter (trend strength)
    df_1d = get_htf_data(prices, '1d')
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
    
    # ATR(10) on 1d
    atr_10_1d = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    # ATR MA(50) for regime filter
    atr_ma_50_1d = pd.Series(atr_10_1d).rolling(window=50, min_periods=50).mean().values
    # High volatility regime: current ATR > MA of ATR
    high_vol_regime = atr_10_1d > atr_ma_50_1d
    
    # Align 1d indicators to 6h timeframe
    highest_20_aligned = align_htf_to_ltf(prices, df_1d, highest_20)  # Note: using 1d highest_20 as placeholder - will fix
    lowest_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_20)    # Need to recalculate - fixing below
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    high_vol_regime_aligned = align_htf_to_ltf(prices, df_1d, high_vol_regime.astype(float))
    
    # Recalculate Donchian on 1d for proper alignment
    highest_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid_1d = (highest_20_1d + lowest_20_1d) / 2.0
    
    # Properly align 1d Donchian levels
    highest_20_1d_aligned = align_htf_to_ltf(prices, df_1d, highest_20_1d)
    lowest_20_1d_aligned = align_htf_to_ltf(prices, df_1d, lowest_20_1d)
    donchian_mid_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for Donchian, 50 for ATR MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(highest_20_1d_aligned[i]) or 
            np.isnan(lowest_20_1d_aligned[i]) or 
            np.isnan(donchian_mid_1d_aligned[i]) or 
            np.isnan(high_vol_regime_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Close breaks above 1d Donchian upper + high vol regime + volume spike
            if (close[i] > highest_20_1d_aligned[i] and 
                high_vol_regime_aligned[i] > 0.5 and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below 1d Donchian lower + high vol regime + volume spike
            elif (close[i] < lowest_20_1d_aligned[i] and 
                  high_vol_regime_aligned[i] > 0.5 and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close reverts to 1d Donchian midpoint
            if close[i] <= donchian_mid_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close reverts to 1d Donchian midpoint
            if close[i] >= donchian_mid_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dATRRegime_VolumeSpike"
timeframe = "6h"
leverage = 1.0
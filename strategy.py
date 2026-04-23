#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Primary timeframe: 4h, HTF: 1d for ATR-based regime detection
- Long: Close breaks above Donchian upper (20-period high) + ATR(1d) < median ATR(1d) (low volatility regime) + volume > 1.5x 20-period avg
- Short: Close breaks below Donchian lower (20-period low) + ATR(1d) < median ATR(1d) (low volatility regime) + volume > 1.5x 20-period avg
- Exit: Close reverts to Donchian midpoint (mean of upper and lower)
- Uses volatility regime filter to avoid false breakouts in high volatility, volume spike for confirmation
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in both bull and bear markets by trading breakouts in low volatility regimes
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
    
    # Volume confirmation: > 1.5x 20-period average (volume spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_20 + lowest_20) / 2.0
    
    # Calculate 1d ATR for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value has no previous close
    
    # ATR(10) on 1d timeframe
    atr_10_1d = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    # Median ATR over 50 periods for regime classification
    median_atr_1d = pd.Series(atr_10_1d).rolling(window=50, min_periods=50).median().values
    
    # Align 1d indicators to 4h timeframe
    atr_10_aligned = align_htf_to_ltf(prices, df_1d, atr_10_1d)
    median_atr_aligned = align_htf_to_ltf(prices, df_1d, median_atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 10, 50)  # Need 20 for Donchian, 10 for ATR, 50 for median
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(atr_10_aligned[i]) or 
            np.isnan(median_atr_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility regime filter: low volatility (ATR < median ATR)
        low_vol_regime = atr_10_aligned[i] < median_atr_aligned[i]
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Close breaks above Donchian upper + low volatility regime + volume spike
            if (close[i] > highest_20[i] and 
                low_vol_regime and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Donchian lower + low volatility regime + volume spike
            elif (close[i] < lowest_20[i] and 
                  low_vol_regime and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close reverts to Donchian midpoint
            if close[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close reverts to Donchian midpoint
            if close[i] >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATRRegime_VolumeSpike"
timeframe = "4h"
leverage = 1.0
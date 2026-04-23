#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume spike confirmation.
- Donchian(20): Upper = 20-period high, Lower = 20-period low (using prior candles)
- Long: Close > Upper Donchian + volume > 1.5x 20-period avg + ATR(14) < ATR(50) (low vol regime)
- Short: Close < Lower Donchian + volume > 1.5x 20-period avg + ATR(14) < ATR(50) (low vol regime)
- Exit: Opposite Donchian breakout or ATR regime shift to high volatility (ATR(14) > 1.5x ATR(50))
- Uses Donchian for structure, volume for conviction, ATR regime for choppy market avoidance
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
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
    
    # Donchian channels (20-period) - using prior close to avoid look-ahead
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # ATR regime filter: ATR(14) < ATR(50) indicates low volatility/choppy regime
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # first bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Low volatility regime: ATR(14) < ATR(50) (avoid choppy markets)
    low_vol_regime = atr_14 < atr_50
    
    # High volatility exit: ATR(14) > 1.5x ATR(50)
    high_vol_exit = atr_14 > 1.5 * atr_50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for ATR50, 20 for Donchian/volume
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr_14[i]) or
            np.isnan(atr_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Close > Upper Donchian + volume confirmation + low vol regime
            if (close[i] > donchian_upper[i] and 
                volume_confirm and 
                low_vol_regime[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close < Lower Donchian + volume confirmation + low vol regime
            elif (close[i] < donchian_lower[i] and 
                  volume_confirm and 
                  low_vol_regime[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < Lower Donchian OR high volatility regime
            if close[i] < donchian_lower[i] or high_vol_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close > Upper Donchian OR high volatility regime
            if close[i] > donchian_upper[i] or high_vol_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_ATRRegime_VolumeSpike"
timeframe = "4h"
leverage = 1.0
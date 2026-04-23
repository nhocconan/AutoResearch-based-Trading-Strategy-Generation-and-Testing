#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume spike confirmation.
- Donchian upper/lower = 20-period high/low on 12h timeframe
- Long: Close > Upper + volume > 2.0x 20-period avg + ATR(14) < ATR(50) (low volatility regime)
- Short: Close < Lower + volume > 2.0x 20-period avg + ATR(14) < ATR(50) (low volatility regime)
- Exit: Opposite breakout (Close < Upper for long, Close > Lower for short) or ATR regime shift (ATR(14) > ATR(50))
- Uses Donchian for structure, volume for conviction, ATR ratio for regime filter (low vol = breakout prone)
- Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull markets (breakouts with momentum) and bear markets (breakouts filtered by low vol regime)
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
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR regime filter: ATR(14) < ATR(50) indicates low volatility (breakout prone)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    low_vol_regime = atr_14 < atr_50  # True when ATR(14) < ATR(50)
    
    # Calculate 12h Donchian channels (20-period)
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for ATR50, 20 for Donchian/volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(donch_upper[i]) or
            np.isnan(donch_lower[i]) or
            np.isnan(atr_14[i]) or
            np.isnan(atr_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Low volatility regime (ATR(14) < ATR(50))
        vol_regime = low_vol_regime[i]
        
        if position == 0:
            # Long: Close > Upper + volume confirmation + low vol regime
            if (close[i] > donch_upper[i] and 
                volume_confirm and 
                vol_regime):
                signals[i] = 0.25
                position = 1
            # Short: Close < Lower + volume confirmation + low vol regime
            elif (close[i] < donch_lower[i] and 
                  volume_confirm and 
                  vol_regime):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < Upper OR high vol regime (ATR(14) > ATR(50))
            if close[i] < donch_upper[i] or not vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close > Lower OR high vol regime (ATR(14) > ATR(50))
            if close[i] > donch_lower[i] or not vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_ATRRegime_VolumeSpike"
timeframe = "12h"
leverage = 1.0
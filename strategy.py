#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Long: Close breaks above Donchian upper(20) + ATR(14) > ATR(50) (high volatility regime) + volume > 1.5x 20-period avg
- Short: Close breaks below Donchian lower(20) + ATR(14) > ATR(50) (high volatility regime) + volume > 1.5x 20-period avg
- Exit: Close crosses Donchian middle (20-period midpoint) for mean reversion
- Uses Donchian channel from primary timeframe for structure, volatility regime filter to avoid choppy markets,
  and volume confirmation to ensure breakout strength
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
- Discrete position sizing: ±0.25 to balance return and minimize fee churn
- Works in bull markets (breakouts with volatility expansion) and bear markets (mean reversion at channel midpoint)
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
    
    # Volume confirmation: > 1.5x 20-period average (volume filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR calculation for volatility regime filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Donchian channel calculation (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for ATR50, 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(atr_14[i]) or
            np.isnan(atr_50[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(donchian_mid[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility regime filter: ATR(14) > ATR(50) (expanding volatility)
        vol_regime = atr_14[i] > atr_50[i]
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Close breaks above Donchian upper + volatility expansion + volume confirmation
            if (close[i] > donchian_high[i] and 
                vol_regime and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Donchian lower + volatility expansion + volume confirmation
            elif (close[i] < donchian_low[i] and 
                  vol_regime and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close crosses below Donchian middle (mean reversion)
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close crosses above Donchian middle (mean reversion)
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATR_VolumeSpike"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Long: price breaks above Donchian(20) high + ATR(14) > ATR(50) (high volatility regime) + volume > 1.5x 20-period avg
- Short: price breaks below Donchian(20) low + ATR(14) > ATR(50) (high volatility regime) + volume > 1.5x 20-period avg
- Exit: price re-enters Donchian(10) range (faster mean reversion) OR ATR regime shifts to low volatility (ATR(14) < ATR(50))
- Uses Donchian for structure, ATR regime to filter choppy markets, volume for conviction
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull (buy breakouts in high vol) and bear (sell breakdowns in high vol)
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
    
    # ATR calculations for regime filter
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    high_close[0] = high_low[0]
    low_close[0] = high_low[0]
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Donchian channels
    donch_20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_10_high = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donch_10_low = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for ATR50, 20 for Donchian20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(atr_14[i]) or
            np.isnan(atr_50[i]) or
            np.isnan(donch_20_high[i]) or
            np.isnan(donch_20_low[i]) or
            np.isnan(donch_10_high[i]) or
            np.isnan(donch_10_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # ATR regime filter: high volatility (ATR14 > ATR50) for breakouts
        high_vol_regime = atr_14[i] > atr_50[i]
        
        if position == 0:
            # Long: price breaks above Donchian20 high + volume confirmation + high vol regime
            if (close[i] > donch_20_high[i] and 
                volume_confirm and 
                high_vol_regime):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian20 low + volume confirmation + high vol regime
            elif (close[i] < donch_20_low[i] and 
                  volume_confirm and 
                  high_vol_regime):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price re-enters below Donchian10 high (faster mean reversion) OR low vol regime
            if close[i] < donch_10_high[i] or not high_vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters above Donchian10 low (faster mean reversion) OR low vol regime
            if close[i] > donch_10_low[i] or not high_vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_ATRRegime_VolumeConfirm"
timeframe = "4h"
leverage = 1.0
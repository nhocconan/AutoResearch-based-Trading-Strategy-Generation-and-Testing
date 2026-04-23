#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume spike.
- Long: price breaks above 20-period high + ATR(14)/ATR(50) > 0.8 (low volatility regime) + volume > 1.5x 20-period avg
- Short: price breaks below 20-period low + ATR(14)/ATR(50) > 0.8 + volume > 1.5x 20-period avg
- Exit: price re-enters 20-period Donchian channel (mean reversion)
- Uses Donchian for structure, ATR regime to avoid high volatility chop, volume for confirmation
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull (buy breakouts) and bear (sell breakdowns) with volatility filter
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
    
    # ATR regime filter: ATR(14) / ATR(50) > 0.8 indicates low volatility (trending regime)
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]  # First bar
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = np.where(atr_50 > 0, atr_14 / atr_50, 0)
    
    # Donchian channels: 20-period high/low
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for ATR50, 20 for Donchian/volume
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(atr_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average) and low volatility regime (ATR ratio > 0.8)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        low_vol_regime = atr_ratio[i] > 0.8
        
        if position == 0:
            # Long: price breaks above Donchian high + volume confirmation + low volatility regime
            if (close[i] > donchian_high[i] and 
                volume_confirm and 
                low_vol_regime):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume confirmation + low volatility regime
            elif (close[i] < donchian_low[i] and 
                  volume_confirm and 
                  low_vol_regime):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price re-enters below Donchian high (mean reversion)
            if close[i] < donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters above Donchian low (mean reversion)
            if close[i] > donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_ATRRegime_VolumeSpike"
timeframe = "4h"
leverage = 1.0
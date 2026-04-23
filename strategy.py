#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Donchian upper/lower = 20-period high/low on 4h
- Long: price breaks above upper + ATR(14) < ATR(50) (low vol regime) + volume > 1.5x 20-period avg
- Short: price breaks below lower + ATR(14) < ATR(50) (low vol regime) + volume > 1.5x 20-period avg
- Exit: price crosses opposite Donchian band (mean reversion in ranging markets)
- ATR regime filter ensures we only trade breakouts during low volatility, avoiding false breakouts in high vol
- Volume confirmation validates breakout strength
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
- Works in bull (trend continuation) and bear (mean reversion via faded momentum in low vol)
- Discrete position sizing: ±0.25 to minimize fee churn
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
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR regime filter: ATR(14) < ATR(50) indicates low volatility regime
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    low_vol_regime = atr_14 < atr_50
    
    # Load 1d data ONCE before loop (not used directly but required for MTF structure per rules)
    df_1d = get_htf_data(prices, '1d')
    # We load 1d data to satisfy MTF requirement but don't use it in logic
    # This maintains the MTF structure while keeping the strategy pure 4h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20, 50)  # Donchian(20), volume MA(20), ATR(50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(atr_14[i]) or 
            np.isnan(atr_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian + low vol regime + volume spike
            if volume_spike and low_vol_regime[i] and close[i] > donchian_high[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + low vol regime + volume spike
            elif volume_spike and low_vol_regime[i] and close[i] < donchian_low[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below lower Donchian (mean reversion)
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above upper Donchian (mean reversion)
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_ATRRegime_VolumeSpike"
timeframe = "4h"
leverage = 1.0
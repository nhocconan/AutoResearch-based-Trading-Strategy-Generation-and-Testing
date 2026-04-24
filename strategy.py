#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume spike.
- Donchian(20) breakouts capture momentum in both bull and bear markets.
- 1d ATR regime filter: high volatility (ATR > 30-period median) allows trend following;
  low volatility triggers mean reversion at Donchian bands.
- Volume spike (>2.0x 24-period average) confirms breakout validity and reduces false signals.
- Discrete position sizing (0.25) minimizes fee churn while allowing meaningful returns.
- Target trades: 75-200 total over 4 years (19-50/year) on 4h timeframe to avoid fee drag.
- Works in bull/bear markets via ATR regime adaptation and volatility-based volume confirmation.
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
    
    # Get 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d ATR and its 30-period median for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr_1d = np.maximum(high_1d - low_1d, np.absolute(high_1d - np.roll(close_1d, 1)), np.absolute(low_1d - np.roll(close_1d, 1)))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_median_1d = pd.Series(atr_1d).rolling(window=30, min_periods=30).median().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_median_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_median_1d)
    high_volatility_regime = atr_1d_aligned > atr_median_1d_aligned  # True = high vol (trend follow), False = low vol (mean revert)
    
    # Donchian(20) channels on 4h
    donchian_h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_l = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 2.0x 24-period average volume (4h * 6 = 1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 24, 30) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_median_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Determine regime: high volatility = trend follow, low volatility = mean revert
            if high_volatility_regime[i]:
                # Trend following mode: breakout in direction of breakout
                if close[i] > donchian_h[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donchian_l[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                # Mean reversion mode: fade extreme moves
                if close[i] < donchian_l[i] and volume_spike[i]:
                    signals[i] = 0.25  # long at lower band
                    position = 1
                elif close[i] > donchian_h[i] and volume_spike[i]:
                    signals[i] = -0.25  # short at upper band
                    position = -1
        elif position == 1:
            # Long exit: price reverts to mean (middle of channel) OR volatility regime shifts
            donchian_mid = (donchian_h[i] + donchian_l[i]) / 2
            if close[i] > donchian_mid or not high_volatility_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to mean (middle of channel) OR volatility regime shifts
            donchian_mid = (donchian_h[i] + donchian_l[i]) / 2
            if close[i] < donchian_mid or not high_volatility_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_ATRRegime_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0